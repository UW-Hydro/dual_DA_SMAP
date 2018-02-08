 
import numpy as np
import pandas as pd
import os
import string
from collections import OrderedDict
import xarray as xr
import datetime as dt
import multiprocessing as mp
import shutil
import scipy.linalg as la
import glob
import xesmf as xe

from tonic.models.vic.vic import VIC, default_vic_valgrind_error_code

import timeit


class VICReturnCodeError(Exception):
    pass


class States(object):
    ''' This class is a VIC states object

    Atributes
    ---------
    ds: <xarray.dataset>
        A dataset of VIC states
    
    Methods
    ---------
    add_gaussian_white_noise_soil_moisture(self, P)
        Add Gaussian noise for all active grid cells
    

    Require
    ---------
    numpy
    xarray
    '''

    
    def __init__(self, ds):
        self.ds = ds
        self.da_EnKF = self.convert_VICstates_to_EnKFstates_sm()
    
    
    def convert_VICstates_to_EnKFstates_sm(self):
        ''' This function extracts all the soil moisture states from the original
            VIC state file ds, and converts to a da with dimension [lat, lon, n],
            where n is the total number of states in EnKF.
        
        Returns
        ----------
        da_sm_states: <xr.DataArray>
            A re-shaped soil moisture DataArray;
            Dimension: [lat, lon, n],
                where len(n) = len(nlayer) * len(veg_class) * len(snow_band)
        '''
             
        # Extract coordinates
        veg_class = self.ds['veg_class']
        snow_band = self.ds['snow_band']
        nlayer = self.ds['nlayer']
        lat = self.ds['lat']
        lon = self.ds['lon']
        
        # Initialize new DataArray
        n = len(veg_class) * len(snow_band) * len(nlayer)
        data = np.empty([len(lat), len(lon), n])
        data[:] = np.nan
        da_EnKF = xr.DataArray(data,
                               coords=[lat, lon, range(n)],
                               dims=['lat', 'lon', 'n'])
        
        # Extract soil moisture states and convert dimension
        EnKF_states = np.rollaxis(self.ds['STATE_SOIL_MOISTURE'].values,
                                  2, 0)  # [nlayer, veg_class, snow_band, lat, lon]
        EnKF_states = EnKF_states.reshape([n, len(lat), len(lon)])
        
        # roll the 'n' dimension to after lat and lon, and fill in da
        EnKF_states = np.rollaxis(EnKF_states, 0, 3) 
        da_EnKF[:] = EnKF_states
        
        # Save as self.da_EnKF
        self.da_EnKF = da_EnKF
        
        return self.da_EnKF
    
    
    def convert_new_EnKFstates_sm_to_VICstates(self, da_EnKF):
        ''' This function converts an EnKF states da (soil moisture states) to the
            VIC states ds (self.ds), with all the other state variables as original in self.ds
            - as a returned ds, withouth changing self.ds.
        
        Parameters
        ----------
        da_EnKF: <xr.DataArray>
            An DataArray of EnKF states, in dimension [lat, lon, n]
            
        Returns
        ----------
        ds: <xr.DataSet>
            A VIC states ds with soil moisture states = da_EnKF, and all the other
            state variables = those in self.ds
        '''
        
        # Extract coordinates
        veg_class = self.ds['veg_class']
        snow_band = self.ds['snow_band']
        nlayer = self.ds['nlayer']
        lat = self.ds['lat']
        lon = self.ds['lon']
        
        # Initialize a new ds for new VIC states
        ds = self.ds.copy(deep=True)
        
        # Convert da_EnKF dimension
        EnKF_states_reshape = da_EnKF.values.reshape(len(lat), len(lon), len(nlayer),
                                                     len(veg_class), len(snow_band))
        EnKF_states_reshape = np.rollaxis(EnKF_states_reshape, 2, 5)  # roll nlayer to the last
        EnKF_states_reshape = np.rollaxis(EnKF_states_reshape, 0, 5)  # put lat and lon to the last
        EnKF_states_reshape = np.rollaxis(EnKF_states_reshape, 0, 5)
        
        # Fill into VIC states ds
        ds['STATE_SOIL_MOISTURE'][:] = EnKF_states_reshape
        
        return ds
        

    def perturb_soil_moisture_Gaussian(self, L, scale_n_nloop, da_max_moist_n,
                                       adjust_negative=True, seed=None):
        ''' Perturb soil moisture states by adding Gaussian noise.
            Each grid cell and soil layer can have different noise magnitude (specified
            by the diagonal of covariance matrix P), but all veg and snow tiles within
            a certain layer and grid cell have the same noise magnitude; within each grid
            cell, noise added to different layers and tiles can be correlated (specified by
            non-diagonal values of covariance matrix P).
            NOTE: this method does not change self

        Parameters
        ----------
        L: <np.array>
            Cholesky decomposed matrix of covariance matrix P of all states:
                            P = L * L.T
            Thus, L * Z (where Z is i.i.d. standard normal random variables of
            dimension [n, n]) is multivariate normal random variables with
            mean zero and covariance matrix P.
            Dimension: [n, n]
        scale_n_nloop: <np.array>
            Standard deviation of noise to add.
            Dimension: [nloop, n] (where nloop = lat * lon)
        da_max_moist_n: <xarray.DataArray>
            Maximum soil moisture for the whole domain and each tile
            [unit: mm]. Soil moistures above maximum after perturbation will
            be reset to maximum value.
            Dimension: [lat, lon, n]
        seed: <int or None>
            Seed for random number generator; this seed will only be used locally
            in this function and will not affect the upper-level code.
            None for not re-assign seed in this function, but using the global seed)
            Default: None

        Returns
        ----------
        ds: <xr.dataset>
            A dataset of VIC states, with soil moisture states perturbed, and all the other
            state variables = those in self.ds
        adjust_negative: <bool>
            Whether or not to adjust negative soil moistures after
            perturbation to zero.
            Default: True (adjust negative to zero)
        '''

        # Extract the number of EnKF states
        n = len(self.da_EnKF['n'])

        # Extract coordinates
        lat = self.ds['lat']
        lon = self.ds['lon']
        veg_class = self.ds['veg_class']
        snow_band = self.ds['snow_band']
        nlayer = self.ds['nlayer']
        
        # Determine total number of loops
        nloop = len(lat) * len(lon)
        n = len(nlayer) * len(veg_class) * len(snow_band)
        
        # Generate N(0, 1) random noise for whole domain and all states
        if seed is None:
            noise = np.array(list(map(
                        np.random.normal,
                        np.zeros(nloop*n),
                        np.ones(nloop*n))))  # [nloop*n]
        else:
            rng = np.random.RandomState(seed)
            noise = np.array(list(map(
                        rng.normal,
                        np.zeros(nloop*n),
                        np.ones(nloop*n))))  # [nloop*n]
        noise = noise.reshape([nloop, n])  # [nloop, n]
        # Apply transformation --> multivariate random noise
        noise = np.array(list(map(
            lambda j: np.dot(L, noise[j, :].reshape([n, 1])),
            range(nloop))))  # [nloop, n, 1]
        noise = noise.reshape([nloop, n])  # [nloop, n]
        # Apply layer perturbation scale
        noise = noise * scale_n_nloop  # [nloop, n]
        noise = noise.reshape([len(lat), len(lon), n])  # [lat, lon, n]

        # Add noise to soil moisture field
        da_perturbed = self.da_EnKF.copy(deep=True)
        da_perturbed[:] += noise

        # Reset negative perturbed soil moistures to zero
        sm_new = da_perturbed.values
        if adjust_negative:
            sm_new[sm_new<0] = 0

        # Reset perturbed soil moistures above maximum to maximum
        max_moist = da_max_moist_n.values
        sm_new[(sm_new>max_moist)] = max_moist[(sm_new>max_moist)]

        # Put into da
        da_perturbed[:] = sm_new

        # Put the perturbed soil moisture states back to VIC states ds
        ds = self.convert_new_EnKFstates_sm_to_VICstates(da_perturbed)
        
        return ds

    
    def update_soil_moisture_states(self, da_K, da_y_meas, da_y_est, R,
                                    da_max_moist_n, adjust_negative=True,
                                    seed=None):
        ''' This function updates soil moisture states for the whole field
            NOTE: this method does not change self
        
        Parameters
        ----------
        da_K: <xr.DataArray>
            Gain K for the whole field
            Dimension: [lat, lon, n, m], where [n, m] is the Kalman gain K
        da_y_meas: <xr.DataArray> [lat, lon, m]
            Measurements at current time
        da_y_est: <xr.DataArray>
            Estimated measurement from before-updated states (y = Hx);
            Dimension: [lat, lon, m]
        R: <float> (for m = 1)
            Measurement error covariance matrix (measurement error ~ N(0, R))
        da_max_moist_n: <xarray.DataArray>
            Maximum soil moisture for the whole domain and each tile
            [unit: mm]. Soil moistures above maximum after perturbation will
            be reset to maximum value.
            Dimension: [lat, lon, n]
        adjust_negative: <bool>
            Whether or not to adjust negative soil moistures after updating
            to zero.
            Default: True (adjust negative to zero)
        seed: <int or None>
            Seed for random number generator; this seed will only be used locally
            in this function and will not affect the upper-level code.
            None for not re-assign seed in this function, but using the global seed)
            Default: None

        Return
        ----------
        da_x_updated: <xr.DataArray>
            Updated soil moisture states
            Dimension: [lat, lon, n]
        da_v: <xr.DataArray>
           Measurement perturbation in the update step
           Dimension: [lat, lon, m]
        '''

        # --- Extract dimensions --- #
        m = len(da_y_est['m'])
        n = len(self.da_EnKF['n'])
        lat_coord = self.da_EnKF['lat']
        lon_coord = self.da_EnKF['lon']

        # --- Initiate updated DataArray for states --- #
        da_x_updated = self.da_EnKF.copy(deep=True)  # [lat, lon, n]

        # --- Update states --- #
        # Determine the total number of loops
        nloop = len(lat_coord) * len(lon_coord)
        # Generate random measurement perturbation
        if seed is None:
            v = np.random.multivariate_normal(
                    np.zeros(m), R,size=nloop).reshape([nloop, m, 1])  # [nloop, m, 1]
        else:
            rng = np.random.RandomState(seed)
            v = rng.multivariate_normal(
                    np.zeros(m), R,size=nloop).reshape([nloop, m, 1])  # [nloop, m, 1]
        # Convert xr.DataArray's to np.array's and straighten lat and lon into nloop
        K = da_K.values.reshape([nloop, n, m])  # [nloop, n, m]
        y_meas = da_y_meas.values.reshape([nloop, m, 1])  # [nloop, m, 1]
        y_est = da_y_est.values.\
                    reshape([nloop, m, 1])  # [nloop, m, 1]
        # Calculate delta = K * (y_meas + v - y_est)
        delta = np.array(list(map(
                    lambda j: np.dot(K[j, :, :],
                                     y_meas[j, :, :] + v[j, :, :] - y_est[j, :, :]),
                    range(nloop))))  # [nloop, n, 1]
        # Reshape delta
        delta = delta.reshape([len(lat_coord), len(lon_coord), n])  # [lat, lon, n]
        # Update states - add delta to orig. states
        da_x_updated[:] += delta

        # --- Reset negative updated soil moistures to zero --- #
        x_new = da_x_updated[:].values
        if adjust_negative:
            x_new[x_new<0] = 0

        # --- Reset updated soil moistures above maximum to maximum --- #
        max_moist = da_max_moist_n.values
        x_new[(x_new>max_moist)] = max_moist[(x_new>max_moist)]

        # --- Put into da --- #
        da_x_updated[:] = x_new

        # --- Save measurement perturbation v to da --- #
        v = v.reshape([len(lat_coord), len(lon_coord), m])  # [lat, lon, m]
        da_v = xr.DataArray(v, coords=[lat_coord, lon_coord, da_y_est['m']],
                            dims=['lat', 'lon', 'm'])

        return(da_x_updated, da_v)


def calculate_sm_noise_to_add_magnitude(vic_history_path, sigma_percent):
    ''' Calculates the standard deviation of Gaussian noise to add for each
        layer and grid cell.

    Parameters
    ----------
    vic_history_path: <str>
        VIC history output file path, typically of openloop run. The range
        of soil moisture for each layer and each grid cell will be used
        to determine soil parameter perturbation level
    sigma_percent: <list>
        List of percentage of the soil moisture range value to perturb.
        Each value of sigma_percent will be used as the standard deviation of
        the Gaussian noise added (e.g., sigma_percent = 5 for 5% of soil 
        moisture range for perturbation).
        Each value in the list is for one soil layer (from up to bottom)

    Returns
    ----------
    da_scale: <xr.DataArray>
        Standard deviation of noise to add
        Dimension: [nlayer, lat, lon]
    '''

    # Load VIC history file and extract soil moisture var
    ds_hist = xr.open_dataset(vic_history_path)
    da_sm = ds_hist['OUT_SOIL_MOIST']
    nlayer = da_sm['nlayer'].values  # coordinate of soil layers

    # Calculate range of soil moisture
    da_range = da_sm.max(dim='time') - da_sm.min(dim='time')

    # Calculate standard devation of noise to add
    da_scale = da_range.copy(deep=True)
    da_scale[:] = np.nan  # [nlayer, lat, lon]
    for i, layer in enumerate(nlayer):
        da_scale.loc[layer, :, :] = da_range.loc[layer, :, :].values\
                                    * sigma_percent[i] / 100.0

    return da_scale


def calculate_cholesky_L(n, corrcoef):
    ''' Calculates covariance matrix for sm state perturbation for one grid cell.

    Parameters
    ----------
    n: <int>
        Total number of EnKF states
    corrcoef: <float>
        Correlation coefficient across different tiles and soil layers

    Returns
    ----------
    L: <np.array>
        Cholesky decomposed matrix of covariance matrix P of all states:
                        P = L * L.T
        Thus, L * Z (where Z is i.i.d. standard normal random variables of
        dimension [n, n]) is multivariate normal random variables with
        mean zero and covariance matrix P.
        Dimension: [n, n]
    '''

    # Calculate P as an correlation coefficient matrix
    P = np.identity(n)
    P[P==0] = corrcoef  # [n, n]

    # Calculate L (Cholesky decomposition: P = L * L.T)
    L = la.cholesky(P).T  # [n, n]

    return L


def calculate_scale_n(layer_scale, nveg, nsnow):
    ''' Calculates covariance matrix for sm state perturbation for one grid cell.

    Parameters
    ----------
    layer_scale: <np.array>
        An array of standard deviation of noise to add for each layer
        Dimension: [nlayer]
    nveg: <int>
        Number of veg classes
    nsnow: <int>
        Number of snow bands

    Returns
    ----------
    scale_n: <np.array>
        Standard deviation of noise to add for a certain grid cell.
        Dimension: [n]
    '''

    nlayer = len(layer_scale)
    n = nlayer * nveg * nsnow

    # Calculate scale_n
    scale_n = np.repeat(layer_scale, nveg*nsnow)  # [n]

    return scale_n


def calculate_scale_n_whole_field(da_scale, nveg, nsnow):
    ''' Calculates covariance matrix for sm state perturbation for one grid cell.

    Parameters
    ----------
    da_scale: <xr.DataArray>
        Standard deviation of noise to add
        Dimension: [nlayer, lat, lon]
    nveg: <int>
        Number of veg classes
    nsnow: <int>
        Number of snow bands

    Returns
    ----------
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon)
        
    Require
    ----------
    calculate_sm_noise_to_add_covariance_matrix
    '''

    # Extract coordinates
    lat = da_scale['lat']
    lon = da_scale['lon']
    nlayer = da_scale['nlayer']
    
    # Determine total number of loops
    nloop = len(lat) * len(lon)
    n = len(nlayer) * nveg * nsnow
    
    # Convert da_scale to np.array and straighten lat and lon into nloop
    scale = da_scale.values.reshape([len(nlayer), nloop])  # [nlayer, nloop]
    scale = np.transpose(scale)  # [nloop, nlayer]
    scale[np.isnan(scale)] = 0  # set inactive cell to zero
    
    # Calculate scale_n for the whole field
    scale_n_nloop = np.array(list(map(calculate_scale_n,
                                      scale,
                                      np.repeat(nveg, nloop),
                                      np.repeat(nsnow, nloop))))  # [nloop, n]

    return scale_n_nloop


class Forcings(object):
    ''' This class is a VIC forcing object

    Atributes
    ---------
    ds: <xarray.dataset>
        A dataset of VIC forcings
    lat_len: <int>
        Length of lat dimension
    lon_len: <int>
        Length of lon dimension
    time_len: <int>
        Length of time dimension
    
    Methods
    ---------
    '''
    
    def __init__(self, ds):
        self.ds = ds
        self.lat_len = len(ds['lat'])
        self.lon_len = len(ds['lon'])
        self.time_len = len(ds['time'])
        self.clean_up_time()
    
    
    def clean_up_time(self):
        ''' Clean up time variable'''
        
        self.ds['time'] = pd.to_datetime(self.ds['time'].values)
        
    
    def perturb_prec_lognormal(self, varname, std=1, phi=0, seed=None):
        ''' Perturb precipitation forcing data
        
        Parameters
        ----------
        varname: <str>
            Precipitation variable name in the forcing ds
        std: <float>
            Standard deviation of the multiplier (which is log-normal distributed);
            Note: here sigma is the real standard deviation of the multiplier, not
            the standard deviation of the underlying normal distribution! (and the
            real mean of the multiplier is 1)
        phi: <float>
            Parameter in AR(1) process. Default: 0 (no autocorrelation).
        seed: <int or None>
            Seed for random number generator; this seed will only be used locally
            in this function and will not affect the upper-level code.
            None for not re-assign seed in this function, but using the global seed)
            Default: None
        '''
        
        # --- Calculate mu and sigma for the lognormal distribution --- #
        # (here mu and sigma are mean and std of the underlying normal dist.)
        mu = -0.5 * np.log(np.square(std)+1)
        sigma = np.sqrt(np.log(np.square(std)+1))

        # Calculate std of white noise and generate random white noise
        scale = sigma * np.sqrt(1 - phi * phi)
        if seed is None:
            white_noise = np.random.normal(
                            loc=0, scale=scale,
                            size=(self.time_len, self.lat_len, self.lon_len))
        else:
            rng = np.random.RandomState(seed)
            white_noise = rng.normal(
                            loc=0, scale=scale,
                            size=(self.time_len, self.lat_len, self.lon_len))

        # --- AR(1) process --- #
        # Initialize
        ar1 = np.empty([self.time_len, self.lat_len, self.lon_len])
        # Generate data for the first time point (need to double check how to do this!!!!!)
        ar1[0, :, :] = white_noise[0, :, :]
        # Loop over each time point
        for t in range(1, self.time_len):
            ar1[t, :, :] = mu + phi * (ar1[t-1, :, :] - mu) +\
                              white_noise[t, :, :]

        # --- Calculate final noise by taking exp --- #
        noise = np.exp(ar1)
        
        # Add noise to soil moisture field
        ds_perturbed = self.ds.copy(deep=True)
        ds_perturbed[varname][:] *= noise
        
        return ds_perturbed

        
    def replace_prec(self, varname, da_prec):
        ''' Replace prec in the forcing by specified prec data
        
        Parameters
        ----------
        varname: <str>
            Precipitation variable name in the current forcing ds
        da_prec: <xr.DataArray>
            A DataArray of prec to be filled in.
            Must have the same dimensions as the forcing object itselt

        Returns
        ----------
        self.ds: <xr.Dataset>
            A dataset of forcings, with specified prec
        '''

        self.ds[varname][:] = da_prec.values

        return self.ds


class VarToPerturb(object):
    ''' This class is a variable to be perturbed

    Atributes
    ---------
    da: <xarray.DataArray>
        A DataArray of the variable to be perturbed
        Dimension: [time, lat, lon]

    Require
    ---------
    numpy
    '''

    def __init__(self, da):
        self.da = da  # dimension: [time, lat, lon]
        self.lat = self.da['lat']
        self.lon = self.da['lon']
        self.time = self.da['time']

    def add_gaussian_white_noise(self, da_sigma, da_max_values,
                                 adjust_negative=True, seed=None):
        ''' Add Gaussian noise for all active grid cells and all time steps

        Parameters
        ----------
        sigma: <xarray.DataArray>
            Standard deviation of the Gaussian white noise to add, can be spatially different
            for each grid cell (but temporally constant);
            Dimension: [lat, lon]
        da_max_values: <xarray.DataArray>
            Maximum values of variable for the whole domain. Perturbed values
            above maximum will be reset to maximum value.
            Dimension: [lat, lon]
        
        Returns
        ----------
        da_perturbed: <xarray.DataArray>
            Perturbed variable for the whole field
            Dimension: [time, lat, lon]
        adjust_negative: <bool>
            Whether or not to adjust negative variable values after
            adding noise to zero.
            Default: True (adjust negative to zero)
        seed: <int or None>
            Seed for random number generator; this seed will only be used locally
            in this function and will not affect the upper-level code.
            None for not re-assign seed in this function, but using the global seed)
            Default: None
        '''

        # Generate random noise for the whole field
        da_noise = self.da.copy(deep=True)
        da_noise[:] = np.nan
        for lt in self.lat:
            for lg in self.lon:
                sigma = da_sigma.loc[lt, lg].values
                if np.isnan(sigma) == True or sigma <= 0:  # if inactive cell, skip
                    continue
                if seed is None:
                    da_noise.loc[:, lt, lg] = np.random.normal(
                            loc=0, scale=sigma, size=len(self.time))
                else:
                    rng = np.random.RandomState(seed)
                    da_noise.loc[:, lt, lg] = rng.normal(
                            loc=0, scale=sigma, size=len(self.time))
        # Add noise to the original da
        da_perturbed = self.da + da_noise
        # Set negative to zero
        tmp = da_perturbed.values
        if adjust_negative:
            tmp[tmp<0] = 0
        # Set perturbed values above maximum to maximum values
        max_values = da_max_values.values  # [lat, lon]
        for i in range(len(self.lat)):
            for j in range(len(self.lon)):
                tmp[(tmp[:, i, j]>max_values[i, j]), i, j] = max_values[i, j]
        # Put into da
        da_perturbed[:] = tmp
        # Add attrs back
        da_perturbed.attrs = self.da.attrs

        return da_perturbed


def EnKF_VIC(N, start_time, end_time, init_state_nc, L, scale_n_nloop, da_max_moist_n,
             R, da_meas,
             da_meas_time_var, vic_exe, vic_global_template,
             ens_forcing_basedir, ens_forcing_prefix,
             vic_model_steps_per_day, output_vic_global_root_dir,
             output_vic_state_root_dir, output_vic_history_root_dir,
             output_vic_log_root_dir,
             mismatched_grid=False,
             weight_nc=None,
             nproc=1,
             mpi_proc=None, mpi_exe='mpiexec', debug=False, output_temp_dir=None,
             linear_model=False, linear_model_prec_varname=None,
             dict_linear_model_param=None):
    ''' This function runs ensemble kalman filter (EnKF) on VIC (image driver)

    Parameters
    ----------
    N: <int>
        Number of ensemble members
    start_time: <pandas.tslib.Timestamp>
        Start time of EnKF run
    end_time: <pandas.tslib.Timestamp>
        End time of EnKF run
    init_state_nc: <str>
        Initial state netCDF file
    L: <np.array>
        Cholesky decomposed matrix of covariance matrix P of all states:
                        P = L * L.T
        Thus, L * Z (where Z is i.i.d. standard normal random variables of
        dimension [n, n]) is multivariate normal random variables with
        mean zero and covariance matrix P.
        Dimension: [n, n]
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon)
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    R: <np.array>  [m*m]
        Measurement error covariance matrix
    da_meas: <xr.DataArray> [time, lat, lon]
        DataArray of measurements (currently, must be only 1 variable of measurement);
        Measurements should already be truncated (if needed) so that they are all within the
        EnKF run period
    da_meas_time_var: <str>
        Time variable name in da_meas
    vic_exe: <class 'VIC'>
        VIC run class
    vic_global_template: <str>
        VIC global file template
    ens_forcing_basedir: <str>
        Ensemble forcing basedir ('ens_{}' subdirs should be under basedir)
    ens_forcing_prefix: <str>
        Prefix of ensemble forcing filenames under 'ens_{}' subdirs
        'YYYY.nc' will be appended
    vic_model_steps_per_day: <int>
        VIC model steps per day
    output_vic_global_root_dir: <str>
        Directory for VIC global files
    output_vic_state_root_dir: <str>
        Directory for VIC output state files
    output_vic_result_root_dir: <str>
        Directory for VIC output result files
    output_vic_log_root_dir: <str>
        Directory for VIC output log files
    mismatched_grid: <bool>
        Whether the measurement grid mismatches VIC grid.
        Default: False
    weight_nc: <str> (Only needed if mismatched_grid is True)
        Weight netCDF file pre-generated by xESMF and process_weight_file
    nproc: <int>
        Number of processors to use
    mpi_proc: <int or None>
        Number of processors to use for VIC MPI run. None for not using MPI
    mpi_exe: <str>
        Path for MPI exe
    debug: <bool>
        True: output temp files for diagnostics; False: do not output temp files
    output_temp_dir: <str>
        Directory for temp files (for dignostic purpose); only used when
        debug = True
    linear_model: <bool>
        Whether to run a linear model instead of VIC for propagation.
        Default is 'False', which is to run VIC
        Note: some VIC input arguments will not be used if this option is True
    linear_model_prec_varname: <str>
        NOTE: this parameter is only needed if linear_model = True.
        Precip varname in the forcing netCDF files.
    dict_linear_model_param: <dict>
        NOTE: this parameter is only needed if linear_model = True.
        A dict of linear model parameters.
        Keys: 'r1', 'r2', 'r3', 'r12', 'r23
        
    Required
    ----------
    numpy
    pandas
    os

    '''

    # --- Pre-processing and checking inputs ---#
    m = 1  # number of measurements
    n_time = len(da_meas[da_meas_time_var])  # number of measurement time points
    # Determine fraction of each veg/snowband tile in each grid cell
    if not linear_model:
        da_tile_frac = determine_tile_frac(vic_global_template)
        adjust_negative = True
    else:
        da_tile_frac = xr.DataArray(
                np.ones([1, 1, len(da_meas['lat']), len(da_meas['lon'])]),
                coords=[[1], [0], da_meas['lat'], da_meas['lon']],
                dims=['veg_class', 'snow_band', 'lat', 'lon'])
        adjust_negative = False
        
    # If mismatched grids, extract weight info
    if mismatched_grid:
        # Extract VIC domain info
        with open(vic_global_template, 'r') as f:
            vic_global_param = f.read()
        vic_domain_nc = find_global_param_value(vic_global_param, 'DOMAIN')
        da_vic_domain = xr.open_dataset(vic_domain_nc)['mask']
        # Extract weight info
        list_source_ind2D_weight_all = extract_mismatched_grid_weight_info(
            da_vic_domain, da_meas, weight_nc)
    else:
        list_source_ind2D_weight_all = None
    
    # Check whether the run period is consistent with VIC setup
    pass
    # Check whether the time range of da_meas is within run period
    pass

    # --- Setup subdirectories --- #
    out_hist_concat_dir = setup_output_dirs(
                output_vic_history_root_dir,
                mkdirs=['EnKF_ensemble_concat'])['EnKF_ensemble_concat']

    # --- Step 1. Initialize ---#
    init_state_time = start_time
    print('\tGenerating ensemble initial states at ', init_state_time)
    time1 = timeit.default_timer()
    # Load initial state file
    ds_states = xr.open_dataset(init_state_nc)
    class_states = States(ds_states)

    # Determine the number of EnKF states in each VIC grid cell, n
    n = len(class_states.da_EnKF['n'])
    # Set up initial state subdirectories
    init_state_dir_name = 'init.{}_{:05d}'.format(
                                    init_state_time.strftime('%Y%m%d'),
                                    init_state_time.hour*3600+init_state_time.second)
    init_state_dir = setup_output_dirs(
                            output_vic_state_root_dir,
                            mkdirs=[init_state_dir_name])[init_state_dir_name]
    # For each ensemble member, add Gaussian noise to sm states with covariance P,
    # and save each ensemble member states
    if debug:
        debug_dir = setup_output_dirs(
                        output_temp_dir,
                        mkdirs=[init_state_dir_name])[init_state_dir_name]
    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(N):
            seed = np.random.randint(low=100000)
            da_perturbation = perturb_soil_moisture_states_class_input(
                    class_states=class_states,
                    L=L,
                    scale_n_nloop=scale_n_nloop,
                    out_states_nc=os.path.join(init_state_dir,
                                               'state.ens{}.nc'.format(i+1)),
                    da_max_moist_n=da_max_moist_n,
                    adjust_negative=adjust_negative,
                    seed=seed)
            # Save soil moisture perturbation
            if debug:
                ds_perturbation = xr.Dataset({'STATE_SOIL_MOISTURE':
                                             (ds - class_states.ds)\
                                  ['STATE_SOIL_MOISTURE']})
                ds_perturbation.to_netcdf(os.path.join(
                        debug_dir,
                        'perturbation.ens{}.nc').format(i+1))
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(N):
            seed = np.random.randint(low=100000)
            pool.apply_async(
                perturb_soil_moisture_states_class_input,
                (class_states, L, scale_n_nloop,
                 os.path.join(init_state_dir, 'state.ens{}.nc'.format(i+1)),
                 da_max_moist_n, adjust_negative, seed))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()

    time2 = timeit.default_timer()
    print('\t\tTime of perturbing init state: {}'.format(time2-time1))

    # --- Step 2. Propagate (run VIC) until the first measurement time point ---#
    # Initialize dictionary of history file paths for each ensemble member
    dict_ens_list_history_files = {}
    for i in range(N):
        dict_ens_list_history_files['ens{}'.format(i+1)] = []

    # Determine VIC run period
    vic_run_start_time = start_time
    vic_run_end_time = pd.to_datetime(da_meas[da_meas_time_var].values[0]) - \
                       pd.DateOffset(hours=24/vic_model_steps_per_day)
    print('\tPropagating (run VIC) until the first measurement time point ',
          pd.to_datetime(da_meas[da_meas_time_var].values[0]))
    time1 = timeit.default_timer()
    # Set up output states, history and global files directories
    propagate_output_dir_name = 'propagate.{}_{:05d}-{}'.format(
                        vic_run_start_time.strftime('%Y%m%d'),
                        vic_run_start_time.hour*3600+vic_run_start_time.second,
                        vic_run_end_time.strftime('%Y%m%d'))
    out_state_dir = setup_output_dirs(
                            output_vic_state_root_dir,
                            mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
    out_history_dir = setup_output_dirs(
                            output_vic_history_root_dir,
                            mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
    out_global_dir = setup_output_dirs(
                            output_vic_global_root_dir,
                            mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
    out_log_dir = setup_output_dirs(
                            output_vic_log_root_dir,
                            mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
    # Propagate all ensemble members
    if not linear_model:
        propagate_ensemble(
                N,
                start_time=vic_run_start_time,
                end_time=vic_run_end_time,
                vic_exe=vic_exe,
                vic_global_template_file=vic_global_template,
                vic_model_steps_per_day=vic_model_steps_per_day,
                init_state_dir=init_state_dir,
                out_state_dir=out_state_dir,
                out_history_dir=out_history_dir,
                out_global_dir=out_global_dir,
                out_log_dir=out_log_dir,
                ens_forcing_basedir=ens_forcing_basedir,
                ens_forcing_prefix=ens_forcing_prefix,
                nproc=nproc,
                mpi_proc=mpi_proc,
                mpi_exe=mpi_exe)
    else:
        propagate_ensemble_linear_model(
                N,
                start_time=vic_run_start_time,
                end_time=vic_run_end_time,
                lat_coord=da_meas['lat'],
                lon_coord=da_meas['lon'],
                model_steps_per_day=vic_model_steps_per_day,
                init_state_dir=init_state_dir,
                out_state_dir=out_state_dir,
                out_history_dir=out_history_dir,
                ens_forcing_basedir=ens_forcing_basedir,
                ens_forcing_prefix=ens_forcing_prefix,
                prec_varname=linear_model_prec_varname,
                dict_linear_model_param=dict_linear_model_param,
                nproc=nproc)
    # Clean up log dir
    shutil.rmtree(out_log_dir)
    
    # Put output history file paths into dictionary
    for i in range(N):
        dict_ens_list_history_files['ens{}'.format(i+1)].append(os.path.join(
                    out_history_dir, 'history.ens{}.{}-{:05d}.nc'.format(
                            i+1,
                            vic_run_start_time.strftime('%Y-%m-%d'),
                            vic_run_start_time.hour*3600+vic_run_start_time.second)))
    time2 = timeit.default_timer()
    print('\t\tTime of propagation: {}'.format(time2-time1))
    
    # --- Step 3. Run EnKF --- #
    debug_innov_dir = setup_output_dirs(
                    output_temp_dir,
                    mkdirs=['innov'])['innov']
    if debug:
        debug_perturbation_dir = setup_output_dirs(
                            output_temp_dir,
                            mkdirs=['perturbation'])['perturbation']
        debug_update_dir = setup_output_dirs(
                        output_temp_dir,
                        mkdirs=['update'])['update']
    # Initialize
    state_dir_after_prop = out_state_dir

    # Loop over each measurement time point
    for t, time in enumerate(da_meas[da_meas_time_var]):

        # Determine last, current and next measurement time points
        current_time = pd.to_datetime(time.values)
        if t == len(da_meas[da_meas_time_var])-1:  # if this is the last measurement time
            next_time = end_time
        else:  # if not the last measurement time
            next_time = pd.to_datetime(da_meas[da_meas_time_var][t+1].values) - \
                        pd.DateOffset(hours=24/vic_model_steps_per_day)
        print('\tCalculating for ', current_time, 'to', next_time)

        # (1.1) Calculate gain K
        time1 = timeit.default_timer()
        da_x, da_y_est = get_soil_moisture_and_estimated_meas_all_ensemble(
                                N,
                                state_dir=state_dir_after_prop,
                                state_time=current_time,
                                da_tile_frac=da_tile_frac,
                                nproc=nproc)
        if mismatched_grid is False:  # if no mismatch
            da_K = calculate_gain_K_whole_field(da_x, da_y_est, R)
        else:  # if mismatche grid
            list_K, y_est_remapped = calculate_gain_K_whole_field_mismatched_grid(
                da_x, da_y_est, R,
                list_source_ind2D_weight_all,
                weight_nc, da_meas)
        if debug:
            ds_K = xr.Dataset({'K': da_K})
            ds_K.to_netcdf(os.path.join(debug_update_dir,
                                        'K.{}_{:05d}.nc'.format(
                                            current_time.strftime('%Y%m%d'),
                                            current_time.hour*3600+current_time.second)))
        time2 = timeit.default_timer()
        print('\t\tTime of calculating gain K: {}'.format(time2-time1))

        # (1.2) Calculate and save normalized innovation
        time1 = timeit.default_timer()
        if mismatched_grid:
            y_est = y_est_remapped.reshape(
                [len(da_meas['lat']), len(da_meas['lon']),
                 y_est_remapped.shape[1], y_est_remapped.shape[2]])  # [lat, lon, m , N]
            da_y_est = xr.DataArray(
                y_est,
                coords=[da_meas['lat'], da_meas['lon'],
                        range(1, y_est_remapped.shape[1]+1),
                        range(1, y_est_remapped.shape[2]+1)],
                dims=['lat', 'lon', 'm', 'N'])
        da_y_est_ensMean = da_y_est.mean(dim='N')  # [lat, lon, m]
        # Calculate non-normalized innovation
        innov = da_meas.loc[time, :, :, :].values - \
                da_y_est_ensMean.values  # [lat, lon, m]
        da_innov = xr.DataArray(innov, coords=[da_y_est_ensMean['lat'],
                                               da_y_est_ensMean['lon'],
                                               da_y_est_ensMean['m']],
                                dims=['lat', 'lon', 'm'])
        # Normalize innovation
        da_Pyy = da_y_est.var(dim='N', ddof=1)  # [lat, lon, m]
        innov_norm = innov / np.sqrt(da_Pyy.values + R)  # [lat, lon, m]
        da_innov_norm = xr.DataArray(innov_norm,
                                     coords=[da_y_est_ensMean['lat'],
                                             da_y_est_ensMean['lon'],
                                             da_y_est_ensMean['m']],
                                     dims=['lat', 'lon', 'm'])
        # Save normalized innovation to netCDf file
        ds_innov_norm = xr.Dataset({'innov_norm': da_innov_norm})
        ds_innov_norm.to_netcdf(os.path.join(
                debug_innov_dir,
                'innov_norm.{}_{:05d}.nc'.format(
                        current_time.strftime('%Y%m%d'),
                        current_time.hour*3600+current_time.second)))
        time2 = timeit.default_timer()
        print('\t\tTime of calculating innovation: {}'.format(time2-time1))

        # (1.3) Update states for each ensemble member
        # Set up dir for updated states
        time1 = timeit.default_timer()
        updated_states_dir_name = 'updated.{}_{:05d}'.format(
                            current_time.strftime('%Y%m%d'),
                            current_time.hour*3600+current_time.second)
        out_updated_state_dir = setup_output_dirs(
                output_vic_state_root_dir,
                mkdirs=[updated_states_dir_name])[updated_states_dir_name]
        # Update states and save to nc files
        if mismatched_grid is False:
            y_est = da_y_est
            K = da_K
        else:
            y_est = y_est_remapped
            K = list_K
        da_x_updated, da_update_increm = update_states_ensemble(
                y_est, K,
                da_meas.loc[time, :, :, :],
                R,
                state_dir_before_update=state_dir_after_prop,
                state_time=current_time,
                out_vic_state_dir=out_updated_state_dir,
                da_max_moist_n=da_max_moist_n,
                mismatched_grid=mismatched_grid,
                list_source_ind2D_weight_all=list_source_ind2D_weight_all,
                adjust_negative=adjust_negative,
                nproc=nproc)
        if debug:
            # Save update increment to netCDF file
            ds_update_increm = xr.Dataset({'update_increment': da_update_increm})
            ds_update_increm.to_netcdf(os.path.join(
                    debug_update_dir,
                    'update_increm.{}_{:05d}.nc'.format(
                             current_time.strftime('%Y%m%d'),
                             current_time.hour*3600+current_time.second)))
        # Delete propagated states
        shutil.rmtree(state_dir_after_prop)
        time2 = timeit.default_timer()
        print('\t\tTime of updating states: {}'.format(time2-time1))

        # (2) Perturb states
        time1 = timeit.default_timer()
        # Set up perturbed state subdirectories
        pert_state_dir_name = 'perturbed.{}_{:05d}'.format(
                                        current_time.strftime('%Y%m%d'),
                                        current_time.hour*3600+current_time.second)
        pert_state_dir = setup_output_dirs(
                            output_vic_state_root_dir,
                            mkdirs=[pert_state_dir_name])[pert_state_dir_name]
        # Perturb states for each ensemble member
        list_da_perturbation = perturb_soil_moisture_states_ensemble(
                    N,
                    states_to_perturb_dir=out_updated_state_dir,
                    L=L,
                    scale_n_nloop=scale_n_nloop,
                    out_states_dir=pert_state_dir,
                    da_max_moist_n=da_max_moist_n,
                    adjust_negative=adjust_negative,
                    nproc=nproc)
        if debug:
            da_perturbation = xr.concat(list_da_perturbation, dim='N')
            da_perturbation['N'] = range(1, N+1)
            ds_perturbation = xr.Dataset({'soil_moisture_perturbation':
                                          da_perturbation})
            ds_perturbation.to_netcdf(os.path.join(
                        debug_perturbation_dir,
                        'perturbation.{}_{:05d}.nc').format(
                                current_time.strftime('%Y%m%d'),
                                current_time.hour*3600+current_time.second))
        time2 = timeit.default_timer()
        print('\t\tTime of perturbing states: {}'.format(time2-time1))

        # (3) Propagate each ensemble member to the next measurement time point
        # If current_time > next_time, do not propagate (we already reach the end of the simulation)
        if current_time > next_time:
            break
        # --- Propagate to the next time point --- #
        time1 = timeit.default_timer()
        propagate_output_dir_name = 'propagate.{}_{:05d}-{}'.format(
                                            current_time.strftime('%Y%m%d'),
                                            current_time.hour*3600+current_time.second,
                                            next_time.strftime('%Y%m%d'))
        out_state_dir = setup_output_dirs(
                                output_vic_state_root_dir,
                                mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
        out_history_dir = setup_output_dirs(
                                output_vic_history_root_dir,
                                mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
        out_global_dir = setup_output_dirs(
                                output_vic_global_root_dir,
                                mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
        out_log_dir = setup_output_dirs(
                                output_vic_log_root_dir,
                                mkdirs=[propagate_output_dir_name])[propagate_output_dir_name]
        if not linear_model:
            propagate_ensemble(
                    N, start_time=current_time, end_time=next_time,
                    vic_exe=vic_exe,
                    vic_global_template_file=vic_global_template,
                    vic_model_steps_per_day=vic_model_steps_per_day,
                    init_state_dir=pert_state_dir,  # perturbed states as init state
                    out_state_dir=out_state_dir,
                    out_history_dir=out_history_dir,
                    out_global_dir=out_global_dir,
                    out_log_dir=out_log_dir,
                    ens_forcing_basedir=ens_forcing_basedir,
                    ens_forcing_prefix=ens_forcing_prefix,
                    nproc=nproc,
                    mpi_proc=mpi_proc,
                    mpi_exe=mpi_exe)
        else:
            propagate_ensemble_linear_model(
                    N,
                    start_time=current_time,
                    end_time=next_time,
                    lat_coord=da_meas['lat'],
                    lon_coord=da_meas['lon'],
                    model_steps_per_day=vic_model_steps_per_day,
                    init_state_dir=pert_state_dir,  # perturbed states as init state
                    out_state_dir=out_state_dir,
                    out_history_dir=out_history_dir,
                    ens_forcing_basedir=ens_forcing_basedir,
                    ens_forcing_prefix=ens_forcing_prefix,
                    prec_varname=linear_model_prec_varname,
                    dict_linear_model_param=dict_linear_model_param,
                    nproc=nproc)
        # Clean up log dir
        shutil.rmtree(out_log_dir)

        # Put output history file paths into dictionary
        for i in range(N):
            dict_ens_list_history_files['ens{}'.format(i+1)].append(os.path.join(
                    out_history_dir, 'history.ens{}.{}-{:05d}.nc'.format(
                            i+1,
                            current_time.strftime('%Y-%m-%d'),
                            current_time.hour*3600+current_time.second)))
        # Delete perturbed states
        shutil.rmtree(pert_state_dir)
        time2 = timeit.default_timer()
        print('\t\tTime of propagation: {}'.format(time2-time1))

        # Point state directory to be updated to the propagated one
        state_dir_after_prop = out_state_dir

        # --- Concat and delete individual history files for each year --- #
        # (If the end of EnKF run, or the end of a calendar year)
        if (t == (len(da_meas[da_meas_time_var]) - 1)) or \
           (t < (len(da_meas[da_meas_time_var]) - 1) and \
           current_time.year != (next_time + \
           pd.DateOffset(hours=24/vic_model_steps_per_day)).year):
            print('\tConcatenating history files...')
            time1 = timeit.default_timer()
            # Determine history file year
            year = current_time.year
            # Identify history dirs to delete later
            list_dir_to_delete = []
            for f in dict_ens_list_history_files['ens1']:
                list_dir_to_delete.append(os.path.dirname(f))
            set_dir_to_delete = set(list_dir_to_delete)  # remove duplicated dirs
            # --- If nproc == 1, do a regular ensemble loop --- #
            if nproc == 1:
                # Concat for each ensemble member and delete individual files
                for i in range(N):
                    # Concat and clean up
                    list_history_files=dict_ens_list_history_files\
                                       ['ens{}'.format(i+1)]
                    output_file=os.path.join(
                                    out_hist_concat_dir,
                                    'history.ens{}.concat.{}.nc'.format(
                                        i+1, year))
                    concat_clean_up_history_file(list_history_files,
                                                 output_file)
                    # Reset history file list
                    dict_ens_list_history_files['ens{}'.format(i+1)] = []
            # --- If nproc > 1, use multiprocessing --- #
            elif nproc > 1:
                # Set up multiprocessing
                pool = mp.Pool(processes=nproc)
                # Loop over each ensemble member
                for i in range(N):
                    # Concat and clean up
                    list_history_files=dict_ens_list_history_files\
                                       ['ens{}'.format(i+1)]
                    output_file=os.path.join(
                                    out_hist_concat_dir,
                                    'history.ens{}.concat.{}.nc'.format(
                                        i+1, year))
                    pool.apply_async(concat_clean_up_history_file,
                                     (list_history_files, output_file))
                    # Reset history file list
                    dict_ens_list_history_files['ens{}'.format(i+1)] = []
                # --- Finish multiprocessing --- #
                pool.close()
                pool.join()
            time2 = timeit.default_timer()
            print('\t\tTime of propagation: {}'.format(time2-time1))

            # Delete history dirs containing individual files
            time1 = timeit.default_timer()
            for d in set_dir_to_delete:
                shutil.rmtree(d)
            time2 = timeit.default_timer()
            print('\t\tTime of deleting history directories: {}'.format(time2-time1))

    # --- Concat and clean up normalized innovation results --- #
    debug_innov_dir
    time1 = timeit.default_timer()
    print('\tInnovation...')
    list_da = []
    list_file_to_delete = []
    times = da_meas[da_meas_time_var].values
    for time in times:
        t = pd.to_datetime(time)
        # Load data
        fname = '{}/innov_norm.{}_{:05d}.nc'.format(
                    debug_innov_dir, t.strftime('%Y%m%d'),
                    t.hour*3600+t.second)
        da = xr.open_dataset(fname)['innov_norm'].sel(m=1)  # [lat, lon]
        # Put data in array
        list_da.append(da)
        # Add individual file to list to delete
        list_file_to_delete.append(fname)
    # Concat innovation of all times
    da_innov_norm = xr.concat(list_da, dim='time')
    da_innov_norm['time'] = da_meas[da_meas_time_var].values
    # Write to file
    ds_innov_norm = xr.Dataset({'innov_norm': da_innov_norm})
    ds_innov_norm.to_netcdf(
        os.path.join(
            debug_innov_dir,
            'innov_norm.concat.{}_{}.nc'.format(
                    pd.to_datetime(times[0]).year,
                    pd.to_datetime(times[-1]).year)),
        format='NETCDF4_CLASSIC')
    # Delete individule files
    for f in list_file_to_delete:
        os.remove(f)
    time2 = timeit.default_timer()
    print('Time of concatenating innovation: {}'.format(time2-time1))

    # --- Concat and clean up debugging results --- #
    if debug:
        # --- Perturbation --- #
        time1 = timeit.default_timer()
        print('\tConcatenating debugging results - perturbation...')
        list_da = []
        list_file_to_delete = []
        times = da_meas[da_meas_time_var].values
        for time in times:
            t = pd.to_datetime(time)
            # Load data
            fname = '{}/perturbation.{}_{:05d}.nc'.format(
                        debug_perturbation_dir, t.strftime('%Y%m%d'),
                        t.hour*3600+t.second)
            da = xr.open_dataset(fname)['soil_moisture_perturbation']
            # Put data in array
            list_da.append(da)
            # Add individual file to list to delete
            list_file_to_delete.append(fname)
        # Concat all times
        da_concat = xr.concat(list_da, dim='time')
        da_concat['time'] = da_meas[da_meas_time_var].values
        # Write to file
        ds_concat = xr.Dataset({'soil_moisture_perturbation': da_concat})
        ds_concat.to_netcdf(
            os.path.join(
                debug_perturbation_dir,
                'perturbation.concat.{}_{}.nc'.format(
                        pd.to_datetime(times[0]).year,
                        pd.to_datetime(times[-1]).year)),
            format='NETCDF4_CLASSIC')
        # Delete individule files
        for f in list_file_to_delete:
            os.remove(f)
        time2 = timeit.default_timer()
        print('Time of concatenating perturbation: {}'.format(time2-time1))

        # --- Update increment --- #
        time1 = timeit.default_timer()
        print('\tConcatenating debugging results - update increment...')
        list_da = []
        list_file_to_delete = []
        times = da_meas[da_meas_time_var].values
        for time in times:
            t = pd.to_datetime(time)
            # Load data
            fname = '{}/update_increm.{}_{:05d}.nc'.format(
                        debug_update_dir, t.strftime('%Y%m%d'),
                        t.hour*3600+t.second)
            da = xr.open_dataset(fname)['update_increment']
            # Put data in array
            list_da.append(da)
            # Add individual file to list to delete
            list_file_to_delete.append(fname)
        # Concat all times
        da_concat = xr.concat(list_da, dim='time')
        da_concat['time'] = da_meas[da_meas_time_var].values
        # Write to file
        ds_concat = xr.Dataset({'update_increment': da_concat})
        ds_concat.to_netcdf(
            os.path.join(
                debug_update_dir,
                'update_increment.concat.{}_{}.nc'.format(
                        pd.to_datetime(times[0]).year,
                        pd.to_datetime(times[-1]).year)),
            format='NETCDF4_CLASSIC')
        # Delete individule files
        for f in list_file_to_delete:
            os.remove(f)
        time2 = timeit.default_timer()
        print('Time of concatenating update increment: {}'.format(time2-time1))

        # --- Gain K --- #
        time1 = timeit.default_timer()
        print('\tConcatenating debugging results - gain K...')
        list_da = []
        list_file_to_delete = []
        times = da_meas[da_meas_time_var].values
        for time in times:
            t = pd.to_datetime(time)
            # Load data
            fname = '{}/K.{}_{:05d}.nc'.format(
                        debug_update_dir, t.strftime('%Y%m%d'),
                        t.hour*3600+t.second)
            da = xr.open_dataset(fname)['K']
            # Put data in array
            list_da.append(da)
            # Add individual file to list to delete
            list_file_to_delete.append(fname)
        # Concat all times
        da_concat = xr.concat(list_da, dim='time')
        da_concat['time'] = da_meas[da_meas_time_var].values
        # Write to file
        ds_concat = xr.Dataset({'K': da_concat})
        ds_concat.to_netcdf(
            os.path.join(
                debug_update_dir,
                'K.concat.{}_{}.nc'.format(
                        pd.to_datetime(times[0]).year,
                        pd.to_datetime(times[-1]).year)),
            format='NETCDF4_CLASSIC')
        # Delete individule files
        for f in list_file_to_delete:
            os.remove(f)
        time2 = timeit.default_timer()
        print('Time of concatenating gain K: {}'.format(time2-time1))

        # --- Measurement perturbation --- #
        time1 = timeit.default_timer()
        print('\tConcatenating debugging results - meas. perturbation...')
        list_da = []
        list_file_to_delete = []
        times = da_meas[da_meas_time_var].values
        for time in times:
            t = pd.to_datetime(time)
            # Load data
            fname = '{}/meas_perturbation.{}_{:05d}.nc'.format(
                        debug_update_dir, t.strftime('%Y%m%d'),
                        t.hour*3600+t.second)
            da = xr.open_dataset(fname)['meas_perturbation']
            # Put data in array
            list_da.append(da)
            # Add individual file to list to delete
            list_file_to_delete.append(fname)
        # Concat all times
        da_concat = xr.concat(list_da, dim='time')
        da_concat['time'] = da_meas[da_meas_time_var].values
        # Write to file
        ds_concat = xr.Dataset({'meas_perturbation': da_concat})
        ds_concat.to_netcdf(
            os.path.join(
                debug_update_dir,
                'meas_perturbation.concat.{}_{}.nc'.format(
                        pd.to_datetime(times[0]).year,
                        pd.to_datetime(times[-1]).year)),
            format='NETCDF4_CLASSIC')
        # Delete individule files
        for f in list_file_to_delete:
            os.remove(f)
        time2 = timeit.default_timer()
        print('Time of concatenating gain meas perturbation: {}'.format(time2-time1))


def to_netcdf_history_file_compress(ds_hist, out_nc):
    ''' This function saves a VIC-history-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_hist: <xr.Dataset>
        History dataset to save
    out_nc: <str>
        Path of output netCDF file
    '''

    dict_encode = {}
    for var in ds_hist.data_vars:
        # skip variables not starting with "OUT_"
        if var.split('_')[0] != 'OUT':
            continue
        # determine chunksizes
        chunksizes = []
        for i, dim in enumerate(ds_hist[var].dims):
            if dim == 'time':  # for time dimension, chunksize = 1
                chunksizes.append(1)
            else:
                chunksizes.append(len(ds_hist[dim]))
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1,
                            'chunksizes': chunksizes}
    ds_hist.to_netcdf(out_nc,
                      format='NETCDF4',
                      encoding=dict_encode)


def to_netcdf_state_file_compress(ds_state, out_nc):
    ''' This function saves a VIC-state-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_state: <xr.Dataset>
        State dataset to save
    out_nc: <str>
        Path of output netCDF file
    '''

    dict_encode = {}
    for var in ds_state.data_vars:
        if var.split('_')[0] != 'STATE':
            continue
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1}
    ds_state.to_netcdf(out_nc,
                       format='NETCDF4',
                       encoding=dict_encode)


def to_netcdf_forcing_file_compress(ds_force, out_nc, time_dim='time'):
    ''' This function saves a VIC-forcing-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_force: <xr.Dataset>
        Forcing dataset to save
    out_nc: <str>
        Path of output netCDF file
    time_dim: <str>
        Time dimension name in ds_force. Default: 'time'
    '''

    dict_encode = {}
    for var in ds_force.data_vars:
        # determine chunksizes
        chunksizes = []
        for i, dim in enumerate(ds_force[var].dims):
            if dim == time_dim:  # for time dimension, chunksize = 1
                chunksizes.append(1)
            else:
                chunksizes.append(len(ds_force[dim]))
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1,
                            'chunksizes': chunksizes}
    ds_force.to_netcdf(out_nc,
                      format='NETCDF4',
                      encoding=dict_encode)


def concat_clean_up_history_file(list_history_files, output_file):
    ''' This function is for wrapping up history file concat and clean up
        for the use of multiprocessing package; history file output is
        compressed.
    
    Parameters
    ----------
    list_history_files: <list>
        A list of output history files (in order) to concat and delete
    output_file: <str>
        Filename for output concatenated netCDF file

    Requires
    ----------
    xarray
    os
    concat_vic_history_files
    '''

    # --- Concat --- #
    ds_concat = concat_vic_history_files(list_history_files)
    # --- Save concatenated file to netCDF (with compression) --- #
    to_netcdf_history_file_compress(ds_concat, output_file)
    # Clean up individual history files
    for f in list_history_files:
        os.remove(f)


def generate_VIC_global_file(global_template_path, model_steps_per_day,
                             start_time, end_time, init_state, vic_state_basepath,
                             vic_history_file_dir, replace,
                             output_global_basepath):
    ''' This function generates a VIC global file from a template file.
    
    Parameters
    ----------
    global_template_path: <str>
        VIC global parameter template (some parts to be filled in)
    model_steps_per_day: <int>
        VIC model steps per day for model run, runoff run and output
    start_time: <pandas.tslib.Timestamp>
        Model run start time
    end_time: <pandas.tslib.Timestamp>
        Model run end time (the beginning of the last time step)
    init_state: <str>
        A full line of initial state option in the global file.
        E.g., "# INIT_STATE"  for no initial state;
              or "INIT_STATE /path/filename" for an initial state file
    vic_state_basepath: <str>
        Output state name directory and file name prefix.
        None if do not want to output state file.
    vic_history_file_dir: <str>
        Output history file directory
    replace: <collections.OrderedDict>
        An ordered dictionary of globap parameters to be replaced
    output_global_basepath: <str>
        Output global file basepath
        ".<start_time>_<end_date>.nc" will be appended,
            where <start_time> is in '%Y%m%d-%H%S',
                  <end_date> is in '%Y%m%d' (since VIC always runs until the end of a date)
    
    Returns
    ----------
    output_global_file: <str>
        VIC global file path
    
    Require
    ----------
    string
    OrderedDict
    pandas
    '''
    
    # --- Create template string --- #
    with open(global_template_path, 'r') as global_file:
        global_param = global_file.read()

    s = string.Template(global_param)
    
    # --- Fill in global parameter options --- #
    state_time = end_time + pd.DateOffset(days=1/model_steps_per_day)

    global_param = s.safe_substitute(model_steps_per_day=model_steps_per_day,
                                     startyear=start_time.year,
                                     startmonth=start_time.month,
                                     startday=start_time.day,
                                     startsec=start_time.hour*3600+start_time.second,
                                     endyear=end_time.year,
                                     endmonth=end_time.month,
                                     endday=end_time.day,
                                     init_state=init_state,
                                     statename=vic_state_basepath,
                                     stateyear=state_time.year, # save state at the end of end_time time step (end_time is the beginning of that time step)
                                     statemonth=state_time.month,
                                     stateday=state_time.day,
                                     statesec=state_time.hour*3600+state_time.second,
                                     result_dir=vic_history_file_dir)
    
    # --- Replace global parameters in replace --- #
    global_param = replace_global_values(global_param, replace)

    # --- If vic_state_basepath == None, add "#" in front of STATENAME --- #
    if vic_state_basepath is None:
        for i, line in enumerate(global_param):
            if line.split()[0] == 'STATENAME':
                global_param[i] = "# STATENAME"
    
    # --- Write global parameter file --- #
    output_global_file = '{}.{}_{}.txt'.format(
                                output_global_basepath,
                                start_time.strftime('%Y%m%d-%H%S'),
                                end_time.strftime('%Y%m%d'))
    
    with open(output_global_file, mode='w') as f:
        for line in global_param:
            f.write(line)

    return output_global_file


def setup_output_dirs(out_basedir, mkdirs=['results', 'state',
                                            'logs', 'plots']):
    ''' This function creates output directories.
    
    Parameters
    ----------
    out_basedir: <str>
        Output base directory for all output files
    mkdirs: <list>
        A list of subdirectories to make

    Require
    ----------
    os
    OrderedDict
    
    Returns
    ----------
    dirs: <OrderedDict>
        A dictionary of subdirectories
    
    '''

    dirs = OrderedDict()
    for d in mkdirs:
        dirs[d] = os.path.join(out_basedir, d)

    for dirname in dirs.values():
        os.makedirs(dirname, exist_ok=True)

    return dirs


def check_returncode(returncode, expected=0):
    '''check return code given by VIC, raise error if appropriate
    
    Require
    ---------
    tonic.models.vic.vic.default_vic_valgrind_error_code
    class VICReturnCodeError
    '''
    if returncode == expected:
        return None
    elif returncode == default_vic_valgrind_error_code:
        raise VICValgrindError('Valgrind raised an error')
    else:
        raise VICReturnCodeError('VIC return code ({0}) does not match '
                                 'expected ({1})'.format(returncode, expected))


def run_vic_for_multiprocess(vic_exe, global_file, log_dir,
                             mpi_proc=None, mpi_exe=None):
    '''This function is a simple wrapper for calling "run" method under
        VIC class in multiprocessing

    Parameters
    ----------
    vic_exe: <str>
        A VIC class object
    global_file: <str>
        VIC global file path
    log_dir: <str>
        VIC run output log directory
    mpi_proc: <int or None>
        Number of processors to use for VIC MPI run. None for not using MPI
    mpi_exe: <str>
        Path for MPI exe. Only used if mpi_proc is not None

    Require
    ----------
    check_returncode
    '''

    if mpi_proc == None:
        returncode = vic_exe.run(global_file, logdir=log_dir,
                                 **{'mpi_proc': mpi_proc})
        check_returncode(returncode, expected=0)
    else:
        returncode = vic_exe.run(global_file, logdir=log_dir,
                                 **{'mpi_proc': mpi_proc, 'mpi_exe': mpi_exe})
        check_returncode(returncode, expected=0)


def propagate_ensemble(N, start_time, end_time, vic_exe, vic_global_template_file,
                       vic_model_steps_per_day, init_state_dir, out_state_dir,
                       out_history_dir, out_global_dir, out_log_dir,
                       ens_forcing_basedir, ens_forcing_prefix, nproc=1,
                       mpi_proc=None, mpi_exe='mpiexec'):
    ''' This function propagates (via VIC) an ensemble of states to a certain time point.
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    start_time: <pandas.tslib.Timestamp>
        Start time of this propagation run
    end_time: <pandas.tslib.Timestamp>
        End time of this propagation
    vic_exe: <class 'VIC'>
        Tonic VIC class
    vic_global_template_file: <str>
        Path of VIC global file template
    vic_model_steps_per_day: <str>
        VIC option - model steps per day
    init_state_dir: <str>
        Directory of initial states for each ensemble member
        State file names are "state.ens<i>", where <i> is 1, 2, ..., N
    out_state_dir: <str>
        Directory of output states for each ensemble member
        State file names will be "state.ens<i>.xxx.nc", where <i> is 1, 2, ..., N
    out_history_dir: <str>
        Directory of output history files for each ensemble member
        History file names will be "history.ens<i>.nc", where <i> is 1, 2, ..., N
    out_global_dir: <str>
        Directory of output global files for each ensemble member
        Global file names will be "global.ens<i>.txt", where <i> is 1, 2, ..., N
    out_log_dir: <str>
        Directory of output log files for each ensemble member
        Log file names will be "global.ens<i>.xxx", where <i> is 1, 2, ..., N
    ens_forcing_basedir: <str>
        Ensemble forcing basedir ('ens_{}' subdirs should be under basedir)
    ens_forcing_prefix: <str>
        Prefix of ensemble forcing filenames under 'ens_{}' subdirs
        'YYYY.nc' will be appended
    nproc: <int>
        Number of processors to use for parallel ensemble
        Default: 1
    mpi_proc: <int or None>
        Number of processors to use for VIC MPI run. None for not using MPI
        Default: None
    mpi_exe: <str>
        Path for MPI exe. Only used if mpi_proc is not None
        
    Require
    ----------
    OrderedDict
    multiprocessing
    generate_VIC_global_file
    '''

    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(N):
            # Generate VIC global param file
            replace = OrderedDict([('FORCING1', os.path.join(
                                            ens_forcing_basedir,
                                            'ens_{}'.format(i+1),
                                            ens_forcing_prefix)),
                                   ('OUTFILE', 'history.ens{}'.format(i+1))])
            global_file = generate_VIC_global_file(
                                global_template_path=vic_global_template_file,
                                model_steps_per_day=vic_model_steps_per_day,
                                start_time=start_time,
                                end_time=end_time,
                                init_state="INIT_STATE {}".format(
                                        os.path.join(
                                            init_state_dir,
                                            'state.ens{}.nc'.format(i+1))),
                                vic_state_basepath=os.path.join(
                                            out_state_dir,
                                            'state.ens{}'.format(i+1)),
                                vic_history_file_dir=out_history_dir,
                                replace=replace,
                                output_global_basepath=os.path.join(
                                            out_global_dir,
                                            'global.ens{}'.format(i+1)))
            # Run VIC
            run_vic_for_multiprocess(vic_exe, global_file, out_log_dir,
                                     mpi_proc, mpi_exe)
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(N):
            # Generate VIC global param file
            replace = OrderedDict([('FORCING1', os.path.join(
                                            ens_forcing_basedir,
                                            'ens_{}'.format(i+1),
                                            ens_forcing_prefix)),
                                   ('OUTFILE', 'history.ens{}'.format(i+1))])
            global_file = generate_VIC_global_file(
                                global_template_path=vic_global_template_file,
                                model_steps_per_day=vic_model_steps_per_day,
                                start_time=start_time,
                                end_time=end_time,
                                init_state="INIT_STATE {}".format(
                                                os.path.join(init_state_dir,
                                                             'state.ens{}.nc'.format(i+1))),
                                vic_state_basepath=os.path.join(out_state_dir,
                                                                'state.ens{}'.format(i+1)),
                                vic_history_file_dir=out_history_dir,
                                replace=replace,
                                output_global_basepath=os.path.join(
                                            out_global_dir,
                                            'global.ens{}'.format(i+1)))
            # Run VIC
            pool.apply_async(run_vic_for_multiprocess,
                             (vic_exe, global_file, out_log_dir, mpi_proc, mpi_exe))
        
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()


def determine_tile_frac(global_path):
    ''' Determines the fraction of each veg/snowband tile in each grid cell based on VIC
        global and parameter files
    
    Parameters
    ----------
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract snowband and vegparam files/options)
    
    Returns
    ----------
    da_tile_frac: <xr.DataArray>
        Fraction of each veg/snowband in each grid cell for the whole domain
        Dimension: [veg_class, snow_band, lat, lon]
    
    Require
    ----------
    numpy
    xarray
    '''
    
    # --- Load global parameter file --- #
    with open(global_path, 'r') as global_file:
            global_param = global_file.read()
            
    # --- Extract Cv from vegparam file (as defined in the global file) --- #
    param_nc = find_global_param_value(global_param, 'PARAMETERS')   
    ds_param = xr.open_dataset(param_nc, decode_cf=False)
    da_Cv = ds_param['Cv']  # dim: [veg_class, lat, lon]
    lat = da_Cv['lat']
    lon = da_Cv['lon']
    
    # --- Extract snowband info from the global and param files --- #
    SNOW_BAND = find_global_param_value(global_param, 'SNOW_BAND')
    if SNOW_BAND.upper() == 'TRUE':
        n_snowband = len(ds_param['snow_band'])
    else:
        n_snowband = 1
    # Dimension of da_AreaFract: [snowband, lat, lon]
    if n_snowband == 1:  # if only one snowband
        data = np.ones([1, len(lat), len(lon)])
        da_AreaFract = xr.DataArray(data, coords=[[0], lat, lon],
                                    dims=['snow_band', 'lat', 'lon'])
    else:  # if more than one snowband
        da_AreaFract = ds_param['AreaFract']

    # --- Initialize the final DataArray --- #
    veg_class = da_Cv['veg_class']
    snow_band = da_AreaFract['snow_band']
    data = np.empty([len(veg_class), len(snow_band), len(lat), len(lon)])
    data[:] = np.nan
    da_tile_frac = xr.DataArray(data, coords=[veg_class, snow_band, lat, lon],
                                dims=['veg_class', 'snow_band', 'lat', 'lon'])
    
    # --- Calculate fraction of each veg/snowband tile for each grid cell,
    # and fill in da_file_frac --- #
    # Determine the total number of loops
    nloop = len(lat) * len(lon)
    # Convert Cv and AreaFract to np.array and straighten lat and lon into nloop
    Cv = da_Cv.values.reshape([len(veg_class), nloop])  # [nveg, nloop]
    AreaFract = da_AreaFract.values.reshape([len(snow_band), nloop])  # [nsnow, nloop]

    # Multiply Cv and AreaFract for each tile and grid cell
    tile_frac = np.array(list(map(
                    lambda i: np.dot(
                        Cv[:, i].reshape([len(veg_class), 1]),
                        AreaFract[:, i].reshape([1, len(snow_band)])),
                    range(nloop))))  # [nloop, nveg, nsnow]

    # Reshape tile_frac
    tile_frac = np.rollaxis(tile_frac, 0, 3)  # [nveg, nsow, nloop]
    tile_frac = tile_frac.reshape([len(veg_class), len(snow_band), len(lat), len(lon)])

    # Put in da_tile_frac
    da_tile_frac[:] = tile_frac
    
    return da_tile_frac


def get_soil_moisture_and_estimated_meas_all_ensemble(N, state_dir, state_time,
                                                      da_tile_frac, nproc):
    ''' This function extracts soil moisture states from netCDF state files for all ensemble
        members, for all grid cells, veg and snow band tiles.
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    state_dir: <str>
        Directory of state files for each ensemble member
        State file names are "state.ens<i>.xxx.nc", where <i> is 1, 2, ..., N
    state_time: <pd.datetime>
        State time. This is for identifying state file names.
    da_tile_frac: <xr.DataArray>
        Fraction of each veg/snowband in each grid cell for the whole domain
        Dimension: [veg_class, snow_band, lat, lon]
    nproc: <int>
        Number of processors to use for parallel ensemble
        Default: 1

    Returns
    ----------
    da_x: <xr.DataArray>
        Soil moisture states of all ensemble members;
        Dimension: [lat, lon, n, N]
    da_y_est: <xr.DataArray>
        Estimated measurement of all ensemble members (= top-layer soil moisture);
        Dimension: [lat, lon, m, N]
    
    Require
    ----------
    xarray
    os
    States
    '''
    
    # --- Extract dimensions from the first ensemble member --- #
    state_name = 'state.ens1.{}_{:05d}.nc'.format(
                        state_time.strftime('%Y%m%d'),
                        state_time.hour*3600+state_time.second)
    ds = xr.open_dataset(os.path.join(state_dir, state_name))
    # nlayer
    nlayer = ds['nlayer']
    # veg_class
    veg_class = ds['veg_class']
    # snow_band
    snow_band = ds['snow_band']
    # lat
    lat = ds['lat']
    # lon
    lon = ds['lon']
    # number of total states n = len(veg_class) * len(snow_band) * len(nlayer)
    n = len(veg_class) * len(snow_band) * len(nlayer)
    
    # --- Initialize da for states and measurement estimates --- #
    # Initialize states x [lat, lon, n, N]
    data = np.empty([len(lat), len(lon), n, N])
    data[:] = np.nan
    da_x = xr.DataArray(data,
                        coords=[lat, lon, range(n), range(N)],
                        dims=['lat', 'lon', 'n', 'N'])
    # Initialize measurement estimates y_est [lat, lon, m, N]
    data = np.empty([len(lat), len(lon), 1, N])
    data[:] = np.nan
    da_y_est = xr.DataArray(data,
                        coords=[lat, lon, [1], range(N)],
                        dims=['lat', 'lon', 'm', 'N'])
    
    # --- Loop over each ensemble member --- #
    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(N):
            # Load state file
            state_name = 'state.ens{}.{}_{:05d}.nc'.format(
                    i+1,
                    state_time.strftime('%Y%m%d'),
                    state_time.hour*3600+state_time.second)
            ds = xr.open_dataset(os.path.join(state_dir, state_name))
            class_states = States(ds)
            
            # Fill x and y_est data in
            # Fill in states x
            da_x.loc[:, :, :, i] = class_states.da_EnKF
            # Fill in measurement estimates y
            da_y_est[:, :, :, i] = calculate_y_est_whole_field(
                                            class_states.ds['STATE_SOIL_MOISTURE'],
                                            da_tile_frac)
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        results = {}
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(N):
            # Load state file
            state_name = 'state.ens{}.{}_{:05d}.nc'.format(
                    i+1,
                    state_time.strftime('%Y%m%d'),
                    state_time.hour*3600+state_time.second)
            ds = xr.open_dataset(os.path.join(state_dir, state_name))
            class_states = States(ds)
 
            # Fill x and y_est data in
            # Fill in states x
            da_x.loc[:, :, :, i] = class_states.da_EnKF
            # Fill in measurement estimates y
            results[i] = pool.apply_async(
                            calculate_y_est_whole_field,
                            (class_states.ds['STATE_SOIL_MOISTURE'],
                             da_tile_frac))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()
        # --- Get return values --- #
        list_da_perturbation = []
        for i, result in results.items():
            da_y_est[:, :, :, i] = result.get()

    return da_x, da_y_est


def calculate_y_est(x_cell, tile_frac_cell):
    ''' Caclulate estimated measurement y_est = Hx for one grid cell; here y_est is
        calculated as tile-average top-layer soil moisture over the whole grid cell.
    
    Parameters
    ----------
    x_cell: <np.array>
        An array of VIC soil moisture states for a grid cell
        Dimension: [veg_class, snow_band, nlayer]
    tile_frac_cell: <np.array>
        An array of veg/band tile fraction for a grid cell
        Dimension: [veg_class, snow_band]
    
    Returns
    ----------
    y_est: <np.float>
        Estimated measurement for this grid cell
    
    Require
    ----------
    numpy
    '''
    
    # Calculate y_est
    y_est = np.nansum(x_cell[:, :, 0] * tile_frac_cell)
    
    return y_est


def calculate_y_est_whole_field(da_x, da_tile_frac):
    ''' Calculate estimated measurement y_est = Hx for all grid cells.
    
    Parameters
    ----------
    da_x: <xr.DataArray>
        A DataArray of VIC soil moisture states for all grid cells
        Dimension: [veg_class, snow_band, nlayer, lat, lon]
    da_tile_frac: <xr.DataArray>
        Fraction of each veg/snowband in each grid cell for the whole domain
        Dimension: [veg_class, snow_band, lat, lon]
        
    Returns
    ----------
    da_y_est: <xr.DataArray>
        Estimated measurement (= top-layer soil moisture) for all grid cells;
        Dimension: [lat, lon, m]
        
    
    Require
    ----------
    xarray
    numpy
    '''
    
    # --- Extract coords --- #
    lat = da_x['lat']
    lon = da_x['lon']
    veg_class = da_x['veg_class']
    snow_band = da_x['snow_band']
    nlayer = da_x['nlayer']
    
    # --- Initiate da_y_est --- #
    data = np.empty([len(lat), len(lon), 1])
    da_y_est = xr.DataArray(data, coords=[lat, lon, [0]], dims=['lat', 'lon', 'm'])
    
    # --- Calculate y_est for all grid cells --- #
    # Determine the total number of loops
    nloop = len(lat) * len(lon)
    # Convert da_x and da_tile_frac to np.array and straighten lat and lon into nloop
    x = da_x.values.reshape([len(veg_class), len(snow_band),
                            len(nlayer), nloop])  # [nveg, nsnow, nlayer, nloop]
    tile_frac = da_tile_frac.values.reshape([len(veg_class), len(snow_band),
                                             nloop])  # [nveg, nsnow, nloop]
    # Calculate y_est for all grid cells
    y_est = np.array(list(map(
                lambda i: calculate_y_est(x[:, :, :, i],
                                          tile_frac[:, :, i]),
                range(nloop)))).reshape([nloop, 1])  # [nloop, m=1]
    # Reshape y_est
    y_est = y_est.reshape([len(lat), len(lon), 1])  # [lat, lon, m=1]
    # Put in da_y_est
    da_y_est[:] = y_est
    
    return da_y_est


def find_global_param_value(gp, param_name, second_param=False):
    ''' Return the value of a global parameter

    Parameters
    ----------
    gp: <str>
        Global parameter file, read in by read()
    param_name: <str>
        The name of the global parameter to find
    second_param: <bool>
        Whether to read a second value for the parameter (e.g., set second_param=True to
        get the snowband param file path when SNOW_BAND>1)

    Returns
    ----------
    line_list[1]: <str>
        The value of the global parameter
    (optional) line_list[2]: <str>
        The value of the second value in the global parameter file when second_param=True
    '''
    for line in iter(gp.splitlines()):
        line_list = line.split()
        if line_list == []:
            continue
        key = line_list[0]
        if key == param_name:
            if second_param == False:
                return line_list[1]
            else:
                return line_list[1], line_list[2]


def calculate_gain_K(x, y_est, R):
    ''' This function calculates Kalman gain K from ensemble.
    
    Parameters
    ----------
    x: <np.array> [n*N]
        An array of forecasted ensemble states (before updated)
    y_est: <np.array> [m*N]
        An array of forecasted ensemble measurement estimates (before updated);
        (y_est = Hx)
    R: <np.array> [m*m]
        Measurement error covariance matrix
    
    Returns
    ----------
    K: <np.array> [n*m]
        Gain K
    
    Require
    ----------
    numpy
    '''
    
    # Extract number of EnKF states (n) and number of measurements (m)
    n = np.shape(x)[0]
    m = np.shape(y_est)[0]
    
    # Pxy = cov(x, y.transpose); size = [n*m]; divided by (N-1)
    Pxy = np.cov(x, y_est)[:n, n:]
    # Pyy = cov(y, y.transpose); size = [m*m]; divided by (N-1)
    Pyy = np.cov(y_est)
    # K = Pxy * (Pyy)-1
    if m == 1:  # if m = 1
        K = Pxy / (Pyy + R)
    else:  # if m > 1
        K = np.dot(Pxx, np.linalg.inv(Pyy+R))

    return K


def calculate_gain_K_whole_field(da_x, da_y_est, R):
    ''' This function calculates gain K over the whole field.
    
    Parameters
    ----------
    da_x: <xr.DataArray>
        Soil moisture states of all ensemble members;
        As returned from get_soil_moisture_and_estimated_meas_all_ensemble;
        Dimension: [lat, lon, n, N]
    da_y_est: <xr.DataArray>
        Estimated measurement of all ensemble members;
        As returned from get_soil_moisture_and_estimated_meas_all_ensemble;
        Dimension: [lat, lon, m, N]
    R: <np.array> [m*m]
        Measurement error covariance matrix
        
    Returns
    ----------
    da_K: <xr.DataArray>
        Gain K for the whole field
        Dimension: [lat, lon, n, m], where [n, m] is the Kalman gain K
    
    Require
    ----------
    xarray
    calculate_gain_K
    numpy
    '''

    # --- Extract dimensions --- #
    lat_coord = da_x['lat']
    lon_coord = da_x['lon']
    n_coord = da_x['n']
    N_coord = da_x['N']
    m_coord = da_y_est['m']
    
    # --- Initialize da_K --- #
    K = np.empty([len(lat_coord), len(lon_coord), len(n_coord), len(m_coord)])
    K[:] = np.nan
    da_K = xr.DataArray(K,
                        coords=[lat_coord, lon_coord, n_coord, m_coord],
                        dims=['lat', 'lon', 'n', 'm'])
    
    # --- Calculate gain K for the whole field --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Convert da_x and da_y_est to np.array and straighten lat and lon into nloop
    x = da_x.values.reshape([nloop, len(n_coord), len(N_coord)])  # [nloop, n, N]
    y_est = da_y_est.values.reshape([nloop, len(m_coord), len(N_coord)])  # [nloop, m, N]
    # Calculate gain K for the whole field
    K = np.array(list(map(
                lambda i: calculate_gain_K(x[i, :, :], y_est[i, :, :], R),
                range(nloop))))  # [nloop, n, m]
    # Reshape K
    K = K.reshape([len(lat_coord), len(lon_coord), len(n_coord),
                   len(m_coord)])  # [lat, lon, n, m]
    # Put in da_K
    da_K[:] = K

    return da_K


def update_states(da_y_est, da_K, da_meas, R, state_nc_before_update,
                  out_vic_state_nc, da_max_moist_n,
                  adjust_negative=True, seed=None):
    ''' Update the EnKF states for the whole field.
    
    Parameters
    ----------
    da_y_est: <xr.DataArray>
        Estimated measurement from pre-updated states (y = Hx);
        Dimension: [lat, lon, m]
    da_K: <xr.DataArray>
        Gain K for the whole field
        Dimension: [lat, lon, n, m], where [n, m] is the Kalman gain K
    da_meas: <xr.DataArray> [lat, lon, m]
        Measurements at current time
    R: <float> (for m = 1)
        Measurement error covariance matrix (measurement error ~ N(0, R))
    state_nc_before_update: <str>
        VIC state file before update
    out_vic_state_nc: <str>
        Path for saving updated state file in VIC format;
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    adjust_negative: <bool>
        Whether or not to adjust negative soil moistures after update
        to zero.
        Default: True (adjust negative to zero)
    seed: <int or None>
        Seed for random number generator; this seed will only be used locally
        in this function and will not affect the upper-level code.
        None for not re-assign seed in this function, but using the global seed)
        Default: None
    
    Returns
    ----------
    da_x_updated: <xr.DataArray>
        Updated soil moisture states;
        Dimension: [lat, lon, n]
    da_update_increm: <xr.DataArray>
        Update increment of soil moisture states
        Dimension: [lat, lon, n]
    da_v: <xr.DataArray>
        Measurement perturbation in the update step
        Dimension: [lat, lon, m]
    
    Require
    ----------
    numpy
    '''

    # Load VIC states before update for this ensemble member          
    ds = xr.open_dataset(state_nc_before_update)
    # Convert EnKF states back to VIC states
    class_states = States(ds)
    # Update states
    da_x_updated, da_v = class_states.update_soil_moisture_states(
                                        da_K, da_meas,
                                        da_y_est, R,
                                        da_max_moist_n,
                                        adjust_negative, seed)  # [lat, lon, n]
    da_update_increm = da_x_updated - class_states.da_EnKF
    # Convert updated states to VIC states format
    ds_updated = class_states.convert_new_EnKFstates_sm_to_VICstates(da_x_updated)
    # Save VIC states to netCDF file (with compression)
    to_netcdf_state_file_compress(ds_updated, out_vic_state_nc)

    return da_x_updated, da_update_increm, da_v


def update_states_ensemble(y_est, K, da_meas, R, state_dir_before_update,
                           state_time, out_vic_state_dir, da_max_moist_n,
                           mismatched_grid=False, list_source_ind2D_weight_all=None,
                           adjust_negative=True, nproc=1):
    ''' Update the EnKF states for the whole field for each ensemble member.

    Parameters
    ----------
    y_est: <xr.DataArray> for mismatched_grid=False; <np.array> for mismatched_grid=True
        Estimated measurement from pre-updated states of all ensemble members (y = Hx);
        If mismatched_grid=False:
            xr.DataArray with Dimension: [lat, lon, m, N]
        If mismatched_grid=True:
            Is actually y_est_remapped; y_est remapped to the meas grid.
            Dim: [n_target, m, N]
    K: <xr.DataArray> for mismatched_grid=False; <list> for mismatched_grid=True
        Gain K for the whole field
        If mismatched_grid=False:
            xr.DataArray with dimension: [lat, lon, n, m],
            where [n, m] is the Kalman gain K
        If mismatched_grid=True:
            Is actually list_K;
            A list of K for each meas grid cell (length n_target)
            Each element of the list is a K matrix of dim [n_stacked, m]
            (n_stacked can be different for each meas cell)
    da_meas: <xr.DataArray> [lat, lon, m]
        Measurements at current time
    R: <float> (for m = 1)
        Measurement error covariance matrix (measurement error ~ N(0, R))
    state_dir_before_update: <str>
        Directory of VIC states before update;
        State file names are: state.ens<i>.<YYYYMMDD>_<SSSSS>.nc,
        where <i> is ensemble member index (1, ..., N),
              <YYYYMMMDD>_<SSSSS> is the current time of the states
    state_time: <pd.datetime>
        State time. This is for identifying state file names.
    output_vic_state_dir: <str>
        Directory for saving updated state files in VIC format;
        State file names will be: state.ens<i>.nc, where <i> is ensemble member index (1, ..., N)
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    mismatched_grid: <bool>
        Whether mismatched measurement and VIC grids. Default: False
    list_source_ind2D_weight_all: <list> (Only needed if mismatched_grid = True)
        Typically returned by extract_mismatched_grid_weight_info().
        list_source_ind2D_weight_all: <list>
        Length of the list: number of meas cells (n_target)
        Each element: a list of ((2D-index), weight) of source cells
        that overlap with ONE target cell
    adjust_negative: <bool>
        Whether or not to adjust negative soil moistures after update
        to zero.
        Default: True (adjust negative to zero)
    nproc: <int>
        Number of processors to use for parallel ensemble
        Default: 1

    Returns
    ----------
    da_x_updated: <xr.DataArray>
        Updated soil moisture states;
        Dimension: [N, lat, lon, n]
    da_update_increm: <xr.DataArray>
        Update increment of soil moisture states
        Dimension: [N, lat, lon, n]
    da_v: <xr.DataArray>
        Measurement perturbation in the update step
        Dimension: [N, lat, lon, m]

    Require
    ----------
    numpy
    '''
    
    # Extract N
    if mismatched_grid is False:
        N = len(y_est['N'])
    else:
        N = y_est.shape[2]

    list_da_update_increm = []  # update increment
    list_da_x_updated = []  # measurement perturbation in the update step

    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(N):
            # Set up parameters
            state_nc_before_update = os.path.join(
                    state_dir_before_update,
                    'state.ens{}.{}_{:05d}.nc'.format(
                            i+1,
                            state_time.strftime('%Y%m%d'),
                            state_time.hour*3600+state_time.second))
            out_vic_state_nc = os.path.join(out_vic_state_dir,
                                            'state.ens{}.nc'.format(i+1))
            # Update states
            seed = np.random.randint(low=100000)
            if mismatched_grid is False:  # if no mismatch
                da_x_updated, da_update_increm = update_states(
                            y_est.loc[:, :, :, i], K, da_meas, R,
                            state_nc_before_update,
                            out_vic_state_nc, da_max_moist_n,
                            adjust_negative, seed)
            else:  # if there is mismatch
                da_x_updated, da_update_increm = update_states_mismatched_grid(
                    y_est[:, :, i], K, da_meas, R, state_nc_before_update,
                    out_vic_state_nc, da_max_moist_n, list_source_ind2D_weight_all,
                    adjust_negative, seed)
            # Put results to list
            list_da_x_updated.append(da_x_updated)
            list_da_update_increm.append(da_update_increm)
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        results = {}
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(N):
            # Set up parameters
            state_nc_before_update = os.path.join(
                    state_dir_before_update,
                    'state.ens{}.{}_{:05d}.nc'.format(
                            i+1,
                            state_time.strftime('%Y%m%d'),
                            state_time.hour*3600+state_time.second))
            out_vic_state_nc = os.path.join(out_vic_state_dir,
                                            'state.ens{}.nc'.format(i+1))
            # Update states
            seed = np.random.randint(low=100000)
            if mismatched_grid is False:  # if no mismatch
                results[i] = pool.apply_async(
                                update_states,
                                (y_est.loc[:, :, :, i], K, da_meas, R,
                                state_nc_before_update,
                                out_vic_state_nc, da_max_moist_n,
                                adjust_negative, seed))
            else:  # if there is mismatch
                results[i] = pool.apply_async(
                    update_states_mismatched_grid,
                        (y_est[:, :, i], K, da_meas, R, state_nc_before_update,
                         out_vic_state_nc, da_max_moist_n, list_source_ind2D_weight_all,
                         adjust_negative, seed))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()
        # --- Get return values --- #
        for i, result in results.items():
            da_x_updated, da_update_increm = result.get()
            # Put results to list
            list_da_x_updated.append(da_x_updated)
            list_da_update_increm.append(da_update_increm)

    # --- Put update increment and measurement perturbation of all ensemble
    # members into one da --- #
    da_x_updated = xr.concat(list_da_x_updated, dim='N')
    da_x_updated['N'] = range(1, N+1)
    da_update_increm = xr.concat(list_da_update_increm, dim='N')
    da_update_increm['N'] = range(1, N+1)

    return da_x_updated, da_update_increm


def perturb_forcings_ensemble(N, orig_forcing, dict_varnames, prec_std,
                              prec_phi, out_forcing_basedir):
    ''' Perturb forcings for all ensemble members

    Parameters
    ----------
    N: <int>
        Number of ensemble members
    orig_forcing: <class 'Forcings'>
        Original (unperturbed) VIC forcings
    dict_varnames: <dict>
        A dictionary of forcing names in nc file;
        e.g., {'PREC': 'prcp'; 'AIR_TEMP': 'tas'}
    prec_std: <float>
        Standard deviation of the precipitation perturbing multiplier
    prec_phi: <float>
        Parameter in AR(1) process for precipitation noise.
    out_forcing_basedir: <str>
        Base directory for output perturbed forcings;
        Subdirs "ens_<i>" will be created, where <i> is ensemble index, 1, ..., N
        File names will be: forc.YYYY.nc 

    Require
    ----------
    os
    '''
    
    # Loop over each ensemble member
    for i in range(N):
        # Setup subdir
        subdir = setup_output_dirs(
                    out_forcing_basedir,
                    mkdirs=['ens_{}'.format(i+1)])['ens_{}'.format(i+1)]

        # Perturb PREC
        ds_perturbed = orig_forcing.perturb_prec_lognormal(
                                            varname=dict_varnames['PREC'],
                                            std=prec_std,
                                            phi=prec_phi)
        # Save to nc file
        for year, ds in ds_perturbed.groupby('time.year'):
            to_netcdf_forcing_file_compress(
                    ds, os.path.join(subdir,
                    'force.{}.nc'.format(year)))


def replace_global_values(gp, replace):
    '''given a multiline string that represents a VIC global parameter file,
       loop through the string, replacing values with those found in the
       replace dictionary'''
    
    gpl = []
    for line in iter(gp.splitlines()):
        line_list = line.split()
        if line_list:
            key = line_list[0]
            if key in replace:
                value = replace.pop(key)
                val = list([str(value)])
            else:
                val = line_list[1:]
            gpl.append('{0: <20} {1}\n'.format(key, ' '.join(val)))

    if replace:
        for key, val in replace.items():
            try:
                value = ' '.join(val)
            except:
                value = val
            gpl.append('{0: <20} {1}\n'.format(key, value))

    return gpl


def perturb_soil_moisture_states_ensemble(N, states_to_perturb_dir,
                                          L, scale_n_nloop,
                                          out_states_dir,
                                          da_max_moist_n,
                                          adjust_negative=True,
                                          nproc=1):
    ''' Perturb all soil_moisture states for each ensemble member
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    states_to_perturb_dir: <str>
        Directory for VIC state files to perturb.
        File names: state.ens<i>.nc, where <i> is ensemble index 1, ..., N
    L: <np.array>
        Cholesky decomposed matrix of covariance matrix P of all states:
                        P = L * L.T
        Thus, L * Z (where Z is i.i.d. standard normal random variables of
        dimension [n, n]) is multivariate normal random variables with
        mean zero and covariance matrix P.
        Dimension: [n, n]
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon)
    out_states_dir: <str>
        Directory for output perturbed VIC state files;
        File names: state.ens<i>.nc, where <i> is ensemble index 1, ..., N
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    adjust_negative: <bool>
        Whether or not to adjust negative soil moistures after
        perturbation to zero.
        Default: True (adjust negative to zero)
    nproc: <int>
        Number of processors to use for parallel ensemble
        Default: 1

    Return
    ----------
    list_da_perturbation: <list>
        A list of amount of perturbation added (a list of each ensemble member)
        Dimension of each da: [veg_class, snow_band, nlayer, lat, lon]
    
    Require
    ----------
    os
    class States
    '''

    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        list_da_perturbation = [] 
        for i in range(N):
            states_to_perturb_nc = os.path.join(
                    states_to_perturb_dir,
                    'state.ens{}.nc'.format(i+1))
            out_states_nc = os.path.join(out_states_dir,
                                         'state.ens{}.nc'.format(i+1))
            seed = np.random.randint(low=100000)
            da_perturbation = perturb_soil_moisture_states(
                                         states_to_perturb_nc, L, scale_n_nloop,
                                         out_states_nc, da_max_moist_n,
                                         adjust_negative, seed)
            list_da_perturbation.append(da_perturbation)
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        results = {}
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(N):
            states_to_perturb_nc = os.path.join(
                    states_to_perturb_dir,
                    'state.ens{}.nc'.format(i+1))
            out_states_nc = os.path.join(out_states_dir,
                                         'state.ens{}.nc'.format(i+1))
            seed = np.random.randint(low=100000)
            results[i] = pool.apply_async(
                    perturb_soil_moisture_states,
                    (states_to_perturb_nc, L, scale_n_nloop,
                     out_states_nc, da_max_moist_n,
                     adjust_negative, seed))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()

        # --- Get return values --- #
        list_da_perturbation = [] 
        for i, result in results.items():
            list_da_perturbation.append(result.get())

    return list_da_perturbation


def propagate(start_time, end_time, vic_exe, vic_global_template_file,
                       vic_model_steps_per_day, init_state_nc, out_state_basepath,
                       out_history_dir, out_history_fileprefix,
                       out_global_basepath, out_log_dir,
                       forcing_basepath, mpi_proc=None, mpi_exe='mpiexec', delete_log=True):
    ''' This function propagates (via VIC) from an initial state (or no initial state)
        to a certain time point.

    Parameters
    ----------
    start_time: <pandas.tslib.Timestamp>
        Start time of this propagation run
    end_time: <pandas.tslib.Timestamp>
        End time of this propagation
    vic_exe: <class 'VIC'>
        Tonic VIC class
    vic_global_template_file: <str>
        Path of VIC global file template
    vic_model_steps_per_day: <str>
        VIC option - model steps per day
    init_state_nc: <str>
        Initial state netCDF file; None for no initial state
    out_state_basepath: <str>
        Basepath of output states; ".YYYYMMDD_SSSSS.nc" will be appended.
        None if do not want to output state file
    out_history_dir: <str>
        Directory of output history files
    out_history_fileprefix: <str>
        History file prefix
    out_global_basepath: <str>
        Basepath of output global files; "YYYYMMDD-HHS_YYYYMMDD.txt" will be appended
    out_log_dir: <str>
        Directory for output log files
    forcing_basepath: <str>
        Forcing basepath. <YYYY.nc> will be appended
    mpi_proc: <int or None>
        Number of processors to use for VIC MPI run. None for not using MPI
        Default: None
    mpi_exe: <str>
        Path for MPI exe. Only used if mpi_proc is not None
    delete_log: <bool>
        Whether to delete all log files in the log directory after running.
        Default: True

    Require
    ----------
    OrderedDict
    generate_VIC_global_file
    check_returncode
    '''

    # Generate VIC global param file
    replace = OrderedDict([('FORCING1', forcing_basepath),
                           ('OUTFILE', out_history_fileprefix)])
    global_file = generate_VIC_global_file(
                        global_template_path=vic_global_template_file,
                        model_steps_per_day=vic_model_steps_per_day,
                        start_time=start_time,
                        end_time=end_time,
                        init_state='#INIT_STATE' if init_state_nc is None
                                   else 'INIT_STATE {}'.format(init_state_nc),
                        vic_state_basepath=out_state_basepath,
                        vic_history_file_dir=out_history_dir,
                        replace=replace,
                        output_global_basepath=out_global_basepath)
    
    # Run VIC
    returncode = vic_exe.run(global_file, logdir=out_log_dir,
                             **{'mpi_proc': mpi_proc, 'mpi_exe': mpi_exe})
    check_returncode(returncode, expected=0)

    # Delete log files (to save space)
    if delete_log:
        for f in glob.glob(os.path.join(out_log_dir, "*")):
            os.remove(f)


def propagate_linear_model(start_time, end_time, lat_coord, lon_coord,
                       model_steps_per_day, init_state_nc, out_state_basepath,
                       out_history_dir, out_history_fileprefix,
                       forcing_basepath, prec_varname, dict_linear_model_param):
    ''' This function propagates (via VIC) from an initial state (or no
        initial state) to a certain time point for the whole domain.
        The linear model assumes 3 soil layers and only allows 1 tile per
        grid cell (i.e., nveg=1 and nsnow=1). Specifically, for each grid cell,
        the model takes the following form:
                    sm(t+1) = r * sm(t) + P(t)
        where sm(t) is an array of dim [3, 1], r is a constant matrix of
        dim [3, 3], and P(t) is input precipitation and is an array of dim
        [3, 1].

    Parameters
    ----------
    start_time: <pandas.tslib.Timestamp>
        Start time of this propagation run
    end_time: <pandas.tslib.Timestamp>
        End time of this propagation
    lat_coord: <list/xr.coord>
        Latitude coordinates
    lon_coord: <list/xr.coord>
        Longitude coordinates
    model_steps_per_day: <str>
        VIC option - model steps per day
    init_state_nc: <str>
        Initial state netCDF file; None for no initial state
    out_state_basepath: <str>
        Basepath of output states; ".YYYYMMDD_SSSSS.nc" will be appended
    out_history_dir: <str>
        Directory of output history files
    out_history_fileprefix: <str>
        History file prefix
    forcing_basepath: <str>
        Forcing basepath. <YYYY.nc> will be appended
    prec_varname: <str>
        Precip varname in the forcing netCDF files
    dict_linear_model_param: <dict>
        A dict of linear model parameters.
        Keys: 'r1', 'r2', 'r3', 'r12', 'r23'
    '''
    
    # --- Establish linear propagation matrix r[3, 3] --- #
    r1 = dict_linear_model_param['r1']
    r2 = dict_linear_model_param['r2']
    r3 = dict_linear_model_param['r3']
    r12 = dict_linear_model_param['r12']
    r23 = dict_linear_model_param['r23']
    r = np.array([[r1-r12, 0, 0],
                  [r12, r2-r23, 0],
                  [0, r23, r3]])

    # --- Establish a list of time steps --- #
    dt_hour = int(24 / model_steps_per_day)  # delta t in hour
    times = pd.date_range(start_time, end_time, freq='{}H'.format(dt_hour))

    # --- Set initial states [3, lat, lon] --- #
    # If not initial states, set all initial soil moistures to zero
    if init_state_nc is None:
        sm0 = np.zeros([3, len(lat_coord), len(lon_coord)])
    # Otherwise, read from an initial state netCDF file
    else:
        sm0 = xr.open_dataset(init_state_nc)['STATE_SOIL_MOISTURE']\
              [0, 0, :, :, :].values

    # --- Load forcing data --- #
    start_year = start_time.year
    end_year = end_time.year
    list_ds = []
    for year in range(start_year, end_year+1):
        ds = xr.open_dataset(forcing_basepath + str(year) + '.nc')
        list_ds.append(ds)
    ds_force = xr.concat(list_ds, dim='time')
    
    # --- Run the linear model --- #
    sm = np.empty([len(times), 3, len(lat_coord), len(lon_coord)])  # [time, 3, lat, lon]
    for i, t in enumerate(times):
        # Extract prec forcing
        da_prec = ds_force[prec_varname].sel(time=t)  # [lat, lon]
        # Determine the total number of loops
        nloop = len(lat_coord) * len(lon_coord)
        # Convert sm into nloop
        if i == 0:
            sm_init = sm0.reshape([3, nloop])  # [3, nloop]
        else:
            sm_init = sm[i-1, :, :, :].reshape([3, nloop])
        # Convert prec into nloop
        prec = da_prec.values.reshape([nloop])
        prec2 = np.zeros(nloop)
        prec3 = np.zeros(nloop)
        prec = np.array([prec, prec2, prec3])  # [3, nloop]
        # Use linear model operate to calculate sm(t)
        sm_new = np.array(list(map(
                    lambda j: np.dot(r, sm_init[:, j]) + prec[:, j],
                    range(nloop))))  # [nloop, 3]
        # Reshape and roll lat and lon to last # [3, lat, lon]
        sm_new = sm_new.reshape([len(lat_coord), len(lon_coord), 3])
        sm_new = np.rollaxis(sm_new, 0, 3)
        sm_new = np.rollaxis(sm_new, 0, 3)
        sm_new = sm_new.reshape([3, len(lat_coord), len(lon_coord)])
        # Put sm_new into the final sm matrix
        sm[i, :, :, :] = sm_new
    
    # --- Put simulated sm into history da and save to history file --- #
    da_sm = xr.DataArray(sm,
                         coords=[times, [0, 1, 2], lat_coord, lon_coord],
                         dims=['time', 'nlayer', 'lat', 'lon'])
    ds_hist = xr.Dataset({'OUT_SOIL_MOIST': da_sm})
    ds_hist.to_netcdf(os.path.join(
            out_history_dir,
            out_history_fileprefix + \
            '.{}-{:05d}.nc'.format(start_time.strftime('%Y-%m-%d'),
                               start_time.hour*3600+start_time.second)))
    
    # --- Save state file at the end of the last time step --- #
    state_time = end_time + pd.DateOffset(hours=dt_hour)
    sm_state = sm[-1, :, :, :].reshape([1, 1, 3, len(lat_coord),
                                        len(lon_coord)])  # [1, 1, 3, lat, lon]
    da_sm_state = xr.DataArray(
            sm_state,
            coords=[[1], [0], [0, 1, 2], lat_coord, lon_coord],
            dims=['veg_class', 'snow_band', 'nlayer', 'lat', 'lon'])
    ds_state = xr.Dataset({'STATE_SOIL_MOISTURE': da_sm_state})
    ds_state.to_netcdf(
            out_state_basepath + \
            '.{}_{:05d}.nc'.format(state_time.strftime('%Y%m%d'),
                                  state_time.hour*3600+state_time.second))


def propagate_ensemble_linear_model(N, start_time, end_time, lat_coord,
                                   lon_coord, model_steps_per_day,
                                   init_state_dir, out_state_dir,
                                   out_history_dir,
                                   ens_forcing_basedir, ens_forcing_prefix,
                                   prec_varname, dict_linear_model_param,
                                   nproc=1):
    ''' This function propagates (via VIC) an ensemble of states to a certain time point.
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    start_time: <pandas.tslib.Timestamp>
        Start time of this propagation run
    end_time: <pandas.tslib.Timestamp>
        End time of this propagation
    lat_coord: <list/xr.coord>
        Latitude coordinates
    lon_coord: <list/xr.coord>
        Longitude coordinates
    model_steps_per_day: <str>
        VIC option - model steps per day
    init_state_dir: <str>
        Directory of initial states for each ensemble member
        State file names are "state.ens<i>", where <i> is 1, 2, ..., N
    out_state_dir: <str>
        Directory of output states for each ensemble member
        State file names will be "state.ens<i>.xxx.nc", where <i> is 1, 2, ..., N
    out_history_dir: <str>
        Directory of output history files for each ensemble member
        History file names will be "history.ens<i>.nc", where <i> is 1, 2, ..., N
    ens_forcing_basedir: <str>
        Ensemble forcing basedir ('ens_{}' subdirs should be under basedir)
    ens_forcing_prefix: <str>
        Prefix of ensemble forcing filenames under 'ens_{}' subdirs
        'YYYY.nc' will be appended
    prec_varname: <str>
        Precip varname in the forcing netCDF files
    dict_linear_model_param: <dict>
        A dict of linear model parameters.
        Keys: 'r1', 'r2', 'r3', 'r12', 'r23'
    nproc: <int>
        Number of processors to use for parallel ensemble
        Default: 1
        
    Require
    ----------
    OrderedDict
    multiprocessing
    generate_VIC_global_file
    '''

    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(N):
            # Prepare linear model parameters
            init_state_nc = os.path.join(init_state_dir,
                                         'state.ens{}.nc'.format(i+1))
            out_state_basepath = os.path.join(out_state_dir,
                                              'state.ens{}'.format(i+1))
            out_history_fileprefix = 'history.ens{}'.format(i+1)
            forcing_basepath = os.path.join(
                    ens_forcing_basedir,
                    'ens_{}'.format(i+1), ens_forcing_prefix)
            # Run linear model
            propagate_linear_model(
                    start_time, end_time, lat_coord, lon_coord,
                    model_steps_per_day, init_state_nc, out_state_basepath,
                    out_history_dir, out_history_fileprefix,
                    forcing_basepath, prec_varname, dict_linear_model_param)

    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(N):
            # Prepare linear model parameters
            init_state_nc = os.path.join(init_state_dir,
                                         'state.ens{}.nc'.format(i+1))
            out_state_basepath = os.path.join(out_state_dir,
                                              'state.ens{}'.format(i+1))
            out_history_fileprefix = 'history.ens{}'.format(i+1)
            forcing_basepath = os.path.join(
                    ens_forcing_basedir,
                    'ens_{}'.format(i+1), ens_forcing_prefix)
            # Run linear model
            pool.apply_async(propagate_linear_model,
                             (start_time, end_time, lat_coord, lon_coord,
                              model_steps_per_day, init_state_nc,
                              out_state_basepath, out_history_dir,
                              out_history_fileprefix, forcing_basepath,
                              prec_varname, dict_linear_model_param))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()


def perturb_soil_moisture_states(states_to_perturb_nc, L, scale_n_nloop,
                                 out_states_nc, da_max_moist_n,
                                 adjust_negative=True, seed=None):
    ''' Perturb all soil_moisture states

    Parameters
    ----------
    states_to_perturb_nc: <str>
        Path of VIC state netCDF file to perturb
    L: <np.array>
        Cholesky decomposed matrix of covariance matrix P of all states:
                        P = L * L.T
        Thus, L * Z (where Z is i.i.d. standard normal random variables of
        dimension [n, n]) is multivariate normal random variables with
        mean zero and covariance matrix P.
        Dimension: [n, n]
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon)
    out_states_nc: <str>
        Path of output perturbed VIC state netCDF file
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    adjust_negative: <bool>
        Whether or not to adjust negative soil moistures after
        perturbation to zero.
        Default: True (adjust negative to zero)
    seed: <int or None>
        Seed for random number generator; this seed will only be used locally
        in this function and will not affect the upper-level code.
        None for not re-assign seed in this function, but using the global seed)
        Default: None

    Returns
    ----------
    da_perturbation: <da.DataArray>
        Amount of perturbation added
        Dimension: [veg_class, snow_band, nlayer, lat, lon]

    Require
    ----------
    os
    class States
    '''

    # --- Load in original state file --- #
    class_states = States(xr.open_dataset(states_to_perturb_nc))

    # --- Perturb --- #
    # Perturb
    ds_perturbed = class_states.perturb_soil_moisture_Gaussian(
                            L, scale_n_nloop, da_max_moist_n,
                            adjust_negative, seed)

    # --- Save perturbed state file --- #
    ds_perturbed.to_netcdf(out_states_nc,
                           format='NETCDF4_CLASSIC')

    # --- Return perturbation --- #
    da_perturbation = (ds_perturbed - class_states.ds)['STATE_SOIL_MOISTURE']

    return da_perturbation


def perturb_soil_moisture_states_class_input(class_states, L, scale_n_nloop,
                                 out_states_nc, da_max_moist_n,
                                 adjust_negative=True, seed=None):
    ''' Perturb all soil_moisture states (same as funciton
        perturb_soil_moisture_states except here inputing class_states
        instead of an netCDF path for states to perturb)

    Parameters
    ----------
    class_states: <class States>
        VIC state netCDF file to perturb
    L: <np.array>
        Cholesky decomposed matrix of covariance matrix P of all states:
                        P = L * L.T
        Thus, L * Z (where Z is i.i.d. standard normal random variables of
        dimension [n, n]) is multivariate normal random variables with
        mean zero and covariance matrix P.
        Dimension: [n, n]
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon)
    out_states_nc: <str>
        Path of output perturbed VIC state netCDF file
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    adjust_negative: <bool>
        Whether or not to adjust negative soil moistures after
        perturbation to zero.
        Default: True (adjust negative to zero)
    seed: <int or None>
        Seed for random number generator; this seed will only be used locally
        in this function and will not affect the upper-level code.
        None for not re-assign seed in this function, but using the global seed)
        Default: None

    Returns
    ----------
    da_perturbation: <da.DataArray>
        Amount of perturbation added
        Dimension: [veg_class, snow_band, nlayer, lat, lon]

    Require
    ----------
    os
    class States
    '''

    # --- Perturb --- #
    # Perturb
    ds_perturbed = class_states.perturb_soil_moisture_Gaussian(
                            L, scale_n_nloop, da_max_moist_n,
                            adjust_negative, seed)

    # --- Save perturbed state file --- #
    to_netcdf_state_file_compress(ds_perturbed, out_states_nc)

    # --- Return perturbation --- #
    da_perturbation = (ds_perturbed - class_states.ds)['STATE_SOIL_MOISTURE']

    return da_perturbation


def concat_vic_history_files(list_history_nc):
    ''' Concatenate short-periods of VIC history files into one; if the time period of
        the next file overlaps that of the current file, the next-file values will be
        used
    
    list_history_nc: <list>
        A list of history files to be concatenated, in order
    '''
    
    print('\tConcatenating {} files...'.format(len(list_history_nc)))

    # --- Open all history files --- #
    list_ds = []
    for file in list_history_nc:
        list_ds.append(xr.open_dataset(file))

    # --- Loop over each history file and concatenate --- #
    list_ds_to_concat = []  # list of ds to concat, with no overlapping periods
    for i in range(len(list_ds[:-1])):
        # Determine and truncate data, if needed
        times_current = pd.to_datetime(list_ds[i]['time'].values)  # times of current ds
        times_next = pd.to_datetime(list_ds[i+1]['time'].values)  # times of next ds
        if times_current[-1] >= times_next[0]:  # if overlap, truncate the current ds
            # Minus 2 seconds to avoid resolution issue
            trunc_time_point = times_next[0] - pd.DateOffset(seconds=2) 
            ds = list_ds[i].sel(time=slice(None, '{}T{:02d}:{:02d}:{:02d}'.format(
                                                trunc_time_point.strftime('%Y-%m-%d'),
                                                trunc_time_point.hour,
                                                trunc_time_point.minute,
                                                trunc_time_point.second)))
        else:  # if no overlap, do not truncate
            ds = list_ds[i]
        # Concat to the list
        list_ds_to_concat.append(ds)
        
    # Concat the last period fully to the list
    list_ds_to_concat.append(list_ds[-1])
   
    # Concat all ds's
    ds_concat = xr.concat(list_ds_to_concat, dim='time')
    
    return ds_concat


def calculate_max_soil_moist_domain(global_path):
    ''' Calculates maximum soil moisture for all grid cells and all soil layers (from soil parameters)
    
    Parameters
    ----------
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract soil parameter file info)
        
    Returns
    ----------
    da_max_moist: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each soil layer [unit: mm];
        Dimension: [nlayer, lat, lon]
    
    Require
    ----------
    xarray
    find_global_param_value
    '''
    
     # Load soil parameter file (as defined in global file)
    with open(global_path, 'r') as global_file:
        global_param = global_file.read()
    soil_nc = find_global_param_value(global_param, 'PARAMETERS')
    ds_soil = xr.open_dataset(soil_nc, decode_cf=False)
    
    # Calculate maximum soil moisture for each layer
    # Dimension: [nlayer, lat, lon]
    da_depth = ds_soil['depth']  # [m]
    da_bulk_density = ds_soil['bulk_density']  # [kg/m3]
    da_soil_density = ds_soil['soil_density']  # [kg/m3]
    da_porosity = 1 - da_bulk_density / da_soil_density
    da_max_moist = da_depth * da_porosity * 1000  # [mm]

    return da_max_moist


def convert_max_moist_n_state(da_max_moist, nveg, nsnow):
    ''' Converts da_max_moist of dimension [nlayer, lat, lon] to [lat, lon, n]
        (Max moistures for each tile within a certain grid cell and layer are the same)

    Parameters
    ----------
    da_max_moist: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each soil layer [unit: mm];
        Dimension: [nlayer, lat, lon]
    nveg: <int>
        Number of veg classes
    nsnow: <int>
        Number of snow bands

    Returns
    ----------
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile [unit: mm];
        Dimension: [lat, lon, n]
    '''
 
    # Extract coordinates
    nlayer = da_max_moist['nlayer']
    lat = da_max_moist['lat']
    lon = da_max_moist['lon']
    n = len(nlayer) * nveg * nsnow

    # Roll dimension to [lat, lon, nlayer]
    max_moist = np.rollaxis(da_max_moist.values, 0, 3)  # [lat, lon, nlayer]

    # Repeat data for each cell and layer for nveg*nsnow times, and convert
    # dimension to [lat, lon, n]
    max_moist_new = np.repeat(max_moist, nveg*nsnow).reshape([len(lat), len(lon), n])

    # Put into da
    da_max_moist_n = xr.DataArray(max_moist_new,
                                  coords=[lat, lon, range(n)],
                                  dims=['lat', 'lon', 'n'])

    return da_max_moist_n


def calculate_ensemble_mean_states(list_state_nc, out_state_nc):
    ''' Calculates ensemble-mean of multiple state files

    Parameters
    ----------
    list_state_nc: <list>
        A list of VIC state nc files whose mean to be calculated
    out_state_nc: <str>
        Path of output state netCDF file

    Returns
    ----------
    out_state_nc: <str>
        Path of output state netCDF file (same as input)

    Require
    ----------
    xarray
    '''

    # --- Number of files --- #
    N = len(list_state_nc)

    # --- Calculate ensemble mean (or median) for each state variable --- #
    list_ds = []
    for state_nc in list_state_nc:
        list_ds.append(xr.open_dataset(state_nc))
    ds_mean = list_ds[0].copy(deep=True)
    # STATE_SOIL_MOISTURE - mean
    ds_mean['STATE_SOIL_MOISTURE'] = (sum(list_ds) / N)['STATE_SOIL_MOISTURE']
    # STATE_SOIL_ICE - mean
    ds_mean['STATE_SOIL_ICE'] = (sum(list_ds) / N)['STATE_SOIL_ICE']
    # STATE_CANOPY_WATER - mean
    ds_mean['STATE_CANOPY_WATER'] = (sum(list_ds) / N)['STATE_CANOPY_WATER']
    # STATE_SNOW_AGE - median, integer
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_AGE'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_AGE'][:] = np.nanmedian(ar_var, axis=0).round()
    # STATE_SNOW_MELT_STATE - median, integer
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_MELT_STATE'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_MELT_STATE'][:] = np.nanmedian(ar_var, axis=0).round()
    # STATE_SNOW_COVERAGE - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_COVERAGE'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_COVERAGE'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_WATER_EQUIVALENT - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_WATER_EQUIVALENT'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_WATER_EQUIVALENT'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_SURF_TEMP - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_SURF_TEMP'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_SURF_TEMP'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_SURF_WATER - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_SURF_WATER'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_SURF_WATER'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_PACK_TEMP - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_PACK_TEMP'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_PACK_TEMP'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_PACK_WATER - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_PACK_WATER'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_PACK_WATER'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_DENSITY - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_DENSITY'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_DENSITY'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_COLD_CONTENT - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_COLD_CONTENT'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_COLD_CONTENT'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SNOW_CANOPY - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_SNOW_CANOPY'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_SNOW_CANOPY'][:] = np.nanmedian(ar_var, axis=0)
    # STATE_SOIL_NODE_TEMP - mean
    ds_mean['STATE_SOIL_NODE_TEMP'] = (sum(list_ds) / N)['STATE_SOIL_NODE_TEMP']
    # STATE_FOLIAGE_TEMPERATURE - mean
    ds_mean['STATE_FOLIAGE_TEMPERATURE'] = (sum(list_ds) / N)['STATE_FOLIAGE_TEMPERATURE']
    # STATE_ENERGY_LONGUNDEROUT - mean
    ds_mean['STATE_ENERGY_LONGUNDEROUT'] = (sum(list_ds) / N)['STATE_ENERGY_LONGUNDEROUT']
    # STATE_ENERGY_SNOW_FLUX - median
    list_var = []
    for ds in list_ds:
        list_var.append(ds['STATE_ENERGY_SNOW_FLUX'].values)
    ar_var = np.asarray(list_var)
    ds_mean['STATE_ENERGY_SNOW_FLUX'][:] = np.nanmedian(ar_var, axis=0)

    # Write to output netCDF file
    ds_mean.to_netcdf(out_state_nc, format='NETCDF4_CLASSIC')

    return out_state_nc


def run_vic_assigned_states(start_time, end_time, vic_exe, init_state_nc,
                            dict_assigned_state_nc, global_template,
                            vic_forcing_basepath, vic_model_steps_per_day,
                            output_global_root_dir,
                            output_vic_history_root_dir,
                            output_vic_log_root_dir, mpi_proc=None, mpi_exe=None,
                            delete_log=True):
    ''' Run VIC with assigned initial states and other assigned state files during the simulation time. All VIC runs do not output state file in the end.
    
    Parameters
    ----------
    start_time: <pandas.tslib.Timestamp>
        Start time of VIC run
    end_time: <pandas.tslib.Timestamp>
        End time of VIC run
    vic_exe: <class 'VIC'>
        VIC run class
    init_state_nc: <str>
        Path of initial state netCDF file; None for no initial state
    dict_assigned_state_nc: <OrderedDict>
        An ordered dictionary of state times and nc files after the start time;
        Keys: state times in <pandas.tslib.Timestamp>;
        Items: state netCDF file path in <str>
    global_template: <str>
        VIC global file template
    vic_forcing_basepath: <str>
        VIC forcing netCDF file basepath ('YYYY.nc' will be appended)
    vic_model_steps_per_day: <int>
        VIC model steps per day
    output_global_root_dir: <str>
        Directory for VIC global files
    output_vic_history_root_dir: <str>
        Directory for VIC output history files
    output_vic_log_root_dir: <str>
        Directory for VIC output log files
    mpi_proc: <int or None>
        Number of processors to use for VIC MPI run. None for not using MPI
    mpi_exe: <str>
        Path for MPI exe
    delete_log: <bool>
        Whether to delete all log files in the log directory after running
        Default: True
    
    Returns
    ----------
    list_history_files_concat: <list>
        A list of all output history file paths in order
    
    Require
    ----------
    OrderedDict
    generate_VIC_global_file
    check_returncode
    '''
    
    list_history_files = []  # A list of resulted history file paths
    list_history_files_concat = []  # A list of final concatenated history file paths
    
    # --- Run VIC from start_time to the first assigned state time --- #
    run_start_time = start_time
    run_end_time = list(dict_assigned_state_nc.keys())[0] - \
                   pd.DateOffset(hours=24/vic_model_steps_per_day)
    print('\tRunning VIC from ', run_start_time, 'to', run_end_time)
    propagate(start_time=run_start_time, end_time=run_end_time,
              vic_exe=vic_exe, vic_global_template_file=global_template,
              vic_model_steps_per_day=vic_model_steps_per_day,
              init_state_nc=init_state_nc,
              out_state_basepath=None,
              out_history_dir=output_vic_history_root_dir,
              out_history_fileprefix='history',
              out_global_basepath=os.path.join(output_global_root_dir, 'global'),
              out_log_dir=output_vic_log_root_dir,
              forcing_basepath=vic_forcing_basepath,
              mpi_proc=mpi_proc,
              mpi_exe=mpi_exe,
              delete_log=delete_log)
    list_history_files.append(os.path.join(
                    output_vic_history_root_dir,
                    'history.{}-{:05d}.nc'.format(
                            run_start_time.strftime('%Y-%m-%d'),
                            run_start_time.hour*3600+run_start_time.second)))
    
    # --- Run VIC from each assigned state time to the next (or to end_time) --- #
    for t, time in enumerate(dict_assigned_state_nc.keys()):
        # --- Determine last, current and next measurement time points --- #
        current_time = time
        if t == len(dict_assigned_state_nc)-1:  # if this is the last measurement time
            next_time = end_time
        else:  # if not the last measurement time
            next_time = list(dict_assigned_state_nc.keys())[t+1] - \
                        pd.DateOffset(hours=24/vic_model_steps_per_day)
        # If current_time > next_time, do not propagate (we already reach the end of the simulation)
        if current_time > next_time:
            break
        print('\tRunning VIC from ', current_time, 'to', next_time)
        
        # --- Propagate to the next time from assigned initial states --- #
        state_nc = dict_assigned_state_nc[current_time]
        propagate(start_time=current_time, end_time=next_time,
                  vic_exe=vic_exe, vic_global_template_file=global_template,
                  vic_model_steps_per_day=vic_model_steps_per_day,
                  init_state_nc=state_nc,
                  out_state_basepath=None,
                  out_history_dir=output_vic_history_root_dir,
                  out_history_fileprefix='history',
                  out_global_basepath=os.path.join(output_global_root_dir, 'global'),
                  out_log_dir=output_vic_log_root_dir,
                  forcing_basepath=vic_forcing_basepath,
                  mpi_proc=mpi_proc,
                  mpi_exe=mpi_exe)
        list_history_files.append(os.path.join(
                    output_vic_history_root_dir,
                    'history.{}-{:05d}.nc'.format(
                            current_time.strftime('%Y-%m-%d'),
                            current_time.hour*3600+current_time.second)))
        # --- Concat and delete individual history files for each year --- #
        # (If the end of EnKF run, or the end of a calendar year)
        if (t == (len(dict_assigned_state_nc.keys()) - 1)) or \
           (t < (len(dict_assigned_state_nc.keys()) - 1) and \
           current_time.year != (next_time + \
           pd.DateOffset(hours=24/vic_model_steps_per_day)).year):
            # Determine history file year
            year = current_time.year
            # Concat individual history files
            output_file=os.path.join(
                            output_vic_history_root_dir,
                            'history.concat.{}.nc'.format(year))
            concat_clean_up_history_file(list_history_files,
                                         output_file)
            list_history_files_concat.append(output_file)
            # Reset history file list
            list_history_files = []
    return list_history_files_concat


def rmse(true, est):
    ''' Calculates RMSE of an estimated variable compared to the truth variable

    Parameters
    ----------
    true: <np.array>
        A 1-D array of time series of true values
    est: <np.array>
        A 1-D array of time series of estimated values (must be the same length of true)

    Returns
    ----------
    rmse: <float>
        Root mean square error

    Require
    ----------
    numpy
    '''

    rmse = np.sqrt(sum((est - true)**2) / len(true))
    return rmse


def innov_norm_var(innov, y_est_before_update, R):
    ''' Calculates normalized variance of innovation
    
    Parameters
    ----------
    innov: <np.array>
        A 1-D array of time series of innovation (meas - y_est_before_update),
        averaged over all ensemble members
    y_est_before_update: <np.array>
        A 2-D array of y_est_before_update for all ensemble members
        Dim: [time, N]
    R: <float>
        Measurement error variance
    
    Returns
    ----------
    var_norm: <float>
        Normalized variance
    
    Require
    ----------
    numpy
    '''
    
    # If all nan, return a time series of nan
    if sum(~np.isnan(innov)) == 0:
        return np.nan
    
    # Pyy = cov(y, y.transpose); divided by (N-1)
    Pyy = np.cov(y_est_before_update).diagonal()  # [time]
    
    # Calculate normalized innovation time series
    innov_norm = innov / np.sqrt(Pyy + R)
    
    # Calculate variance
    var_norm = np.var(innov_norm)
    
    return var_norm


def load_nc_and_concat_var_years(basepath, start_year, end_year, dict_vars):
    ''' Loads in netCDF files end with 'YYYY.nc', and for each variable needed,
        concat all years together and return a DataArray
        
        Parameters
        ----------
        basepath: <str>
            Basepath of all netCDF files; 'YYYY.nc' will be appended;
            Time dimension name in the nc files must be 'time'
        start_year: <int>
            First year to load
        end_year: <int>
            Last year to load
        dict_vars: <dict>
            A dict of desired variables and corresponding varname in the
            netCDF files (e.g., {'prec': 'prcp'; 'temp': 'tair'}). The keys in
            dict_vars will be used as keys in the output dict.

        Returns
        ----------
        dict_da: <dict>
            A dict of concatenated xr.DataArrays.
            Keys: desired variables (using the same keys as in input
                  'dict_vars')
            Elements: <xr.DataArray>
    '''

    dict_list = {}
    for var in dict_vars.keys():
        dict_list[var] = []

    # Loop over each year
    for year in range(start_year, end_year+1):
        # Load data for this year
        ds = xr.open_dataset(basepath + '{}.nc'.format(year))
        # Extract each variable needed and put in a list
        for var, varname in dict_vars.items():
            da = ds[varname]
            # Put data of this year in a list
            dict_list[var].append(da)

    # Concat all years for all variables
    dict_da = {}
    for var in dict_vars.keys():
        dict_da[var] = xr.concat(dict_list[var], dim='time')

    return dict_da


def da_3D_to_2D_for_SMART(dict_da_3D, da_mask, time_varname='time'):
    ''' Convert data arrays with dimension [time, lat, lon] to [npixel_active, time]
    
    Parameters
    ----------
    dict_da_3D: <dict>
        A dict of xr.DataArray's with [time, lat, lon] dimension
        Keys: variable name
        Elements: xr.DataArray
    da_mask: <xr.DataArray>
        Domain mask (>0 for active grid cells)
    time_varname: <str>
        Varname for time in all da's in dict_da_3D
    
    Return
    ----------
    dict_array_active: <dict>
        Keys: variable name (same as in input)
        Elements: np.array with dimension [npixel_active, time]
    
    '''
    
    # Extract ntime, nlat and nlon
    da_example = dict_da_3D[list(dict_da_3D.keys())[0]]
    ntime = len(da_example[time_varname])
    nlat = len(da_example['lat'])
    nlon = len(da_example['lon'])
    
    # Reshape domain mask
    mask = da_mask.values
    mask_reshape = mask.reshape(nlat*nlon)
    
    # Convert all da's in dict_da_3D to [npixel_active, nday]
    dict_array_active = {}
    # Loop over each da
    for var, da in dict_da_3D.items():
        # Convert to np.array and reshape to [ntime, ncell]
        array_all_cells = da.values.reshape(ntime, nlat*nlon)
        # Only keep active grid cells to the final dimension [ntime, npixel_active]
        array_active = array_all_cells[:, mask_reshape>0]
        # Transpose dimension to [npixel_active, nday]
        array_active = np.transpose(array_active)
        # Put in final dictionary
        dict_array_active[var] = array_active
    
    return dict_array_active


def da_2D_to_3D_from_SMART(dict_array_2D, da_mask, out_time_varname, out_time_coord):
    ''' Convert data arrays with dimension [time, npixel_active] to [time, lat, lon]

    Parameters
    ----------
    dict_array_2D: <dict>
        Keys: variable name
        Elements: np.array with dimension [time, npixel_active]
    da_mask: <xr.DataArray>
        Domain mask (>0 for active grid cells)
    out_time_varname: <str>
        Varname for time in output xr.DataArray's
    out_time_coord: <list or other types usable for xr coords>
        Time coordinates in output xr.DataArray's

    Return
    ----------
    dict_da_3D: <dict>
        Keys: variable name (same as in input)
        Elements: xr.DataArray with dimension [time, lat, lon]

    '''

    # Extract ndays, nlat and nlon (as well as lat and lon)
    ntime = len(out_time_coord)
    lat = da_mask['lat']
    lon = da_mask['lon']
    nlat = len(lat)
    nlon = len(lon)    

    # Reshape domain mask and identify indices of active pixels
    mask = da_mask.values
    mask_reshape = mask.reshape(nlat*nlon)
    ind_active = np.where(mask_reshape>0)[0]
    
    # Convert all arrays in dict_da_2D to [time, lat, lon]
    dict_da_3D = {}
    # Loop over each da
    for var, array in dict_array_2D.items():
        # Fill in inactive cells (to dimension [time, ncells])
        array_allcells_reshape = np.empty([ntime, nlat*nlon])
        array_allcells_reshape[:] = np.nan
        array_allcells_reshape[:, ind_active] = array
        # Convert to dimension [time, lat, lon]
        array_3D = array_allcells_reshape.reshape([ntime, nlat, nlon])
        # Put in xr.DataArray
        da = xr.DataArray(array_3D, coords=[out_time_coord, lat, lon],
                          dims=[out_time_varname, 'lat', 'lon'])
        # Put in final dictionary
        dict_da_3D[var] = da

    return dict_da_3D


def correct_prec_from_SMART(da_prec_orig, window_size, da_prec_corr_window,
                            start_date):
    ''' Correct (i.e., rescale) original precipitation data based on outputs from
        SMART (which is corrected prec sum in longer time windows).
            - The first window is not corrected (orig. prec used)
            - The last incomplete window, if exists, is not corrected (orig. prec used)
        
    Parameters
    ----------
    da_prec_orig: <xr.DataArray>
        Original prec data to be corrected. Original timestep, typically daily or
        sub-daily.
        Dims: [time, lat, lon]
    window_size: <int>
        Size of each window [unit: day]
    da_prec_corr_window: <xr.DataArray>
        Corrected prec sum in each window
        Dims: [window (index starting from 0), lat, lon]
        NOTE: this is the output from SMART:
                - the first window data is junk (always zeros)
                - Only include complete windows
    start_date: <dt.datetime or pd.datetime>
        Starting date (midnight) of the starting time of all the time series
        
    Returns
    ----------
    da_prec_corrected: <xr.DataArray>
        Corrected prec data to be corrected. Original timestep, consistent with that
        of input da_prec_orig
        Dims: [time, lat, lon]
    '''
    
    # Identify the number of complete windows
    nwindow = len(da_prec_corr_window['window'])
    
    # Sum original prec in each window (the last window may be incomplete)
    da_prec_orig_window = da_prec_orig.resample(
                            dim='time',
                            freq='{}D'.format(window_size),
                            how='sum')
    da_prec_orig_window = da_prec_orig_window.rename({'time': 'window'})
    
    # Loop over each window and rescale original prec (skip first window)
    da_prec_corrected = da_prec_orig.copy(deep=True)
    for i in range(1, nwindow):
        # --- Select out this window period from original prec data --- #
        window_start_date = start_date + pd.DateOffset(days=i*window_size)
        window_end_date = start_date + pd.DateOffset(days=(i+1)*window_size-0.0001)
        da_prec_orig_this_window = da_prec_orig.sel(
                time=slice(window_start_date, window_end_date))
        # --- Rescale --- #
        # (1) Calculate rescaling factors for this window;
        prec_corr = da_prec_corr_window[i, :, :].values  # dim: [lat, lon]
        prec_sum_orig = da_prec_orig_window[i, :, :].values  # dim: [lat, lon]
        # Note: If prec_sum_orig is 0 for a grid cell, scale_factors will be np.inf for
        # that grid cell
        scale_factors = prec_corr / prec_sum_orig
        # (2) Rescale for the orig. prec data (at orig. sub-daily or daily timestep)
        # for this window (here array broadcast is used)
        prec_corr_this_window = da_prec_orig_this_window.values * scale_factors
        # (3) For grid cells where scale_factors == np.inf or scale == np.nan
        # (which indicates orig. prec are all zero for this window and this
        # cell; np.inf indicates non-zero numerator while np.nan indicates
        # zero numerator; np.nan at inactive grid cells as well), fill in with
        # constant corrected prec from SMART
        # Index of zero-orig-prec grid cells
        ind_inf_scale = np.where(np.isinf(scale_factors) + np.isnan(scale_factors))
        # Constant corrected prec to fill in (divided to each orig. timestep)
        const_prec = prec_corr / len(da_prec_orig_this_window['time'])
        # Fill in those zero-orig-prec grid cells
        const_prec_inf_scale = const_prec[ind_inf_scale[0], ind_inf_scale[1]]
        prec_corr_this_window[:, ind_inf_scale[0], ind_inf_scale[1]] = const_prec_inf_scale
        # --- Put in the final da --- #
        da_prec_corrected.sel(time=slice(window_start_date, window_end_date))[:] = \
                prec_corr_this_window
    
    return da_prec_corrected


def rescale_and_save_SMART_prec(da_prec_orig, window_size,
                                da_prec_corr_window, start_date, out_dir, out_prefix):
    ''' Rescale and save SMART-corrected precipitation (one ensemble member)
    Parameters
    ----------
    da_prec_orig: <xr.DataArray>
        Original precipitation
    window_size: <int>
        Window size in SMART
    da_prec_corr_window_ens: <xr.DataArray>
        SMART-corrected window-averaged precipitation
    start_date: <pd.datetime>
        Start date of simulation
    out_dir: <str>
        Output directory for rescaled precipitation
    out_prefix: <str>
        Prefix of the output prec file ("YYYY.nc" will be appended)
    '''

    # Rescale
    da_prec_corrected = correct_prec_from_SMART(
        da_prec_orig,
        window_size,
        da_prec_corr_window,
        start_date)
    # Save to file
    ds_prec_corrected = xr.Dataset({'prec_corrected': da_prec_corrected})
    for year, ds in ds_prec_corrected.groupby('time.year'):
        to_netcdf_forcing_file_compress(
            ds_force=ds,
            out_nc=os.path.join(out_dir, '{}{}.nc'.format(out_prefix, year)))


def extract_mismatched_grid_weight_info(da_vic_domain, da_meas, weight_nc):
    ''' In the case of mismatched measurement and VIC grids, this
    function extracts all the weight info for remapping.
    
    Parameters
    ----------
    da_vic_domain: <xr.DataArray>
        VIC mask file; 1/0 for active/inactive cells.
    da_meas: <xr.DataArray>
        Measurement data. Only used for extracting lat lon info.
    weight_nc: <str>
        Weight file, pre-calculated by xESMF and process_weight_file()
    
    Return
    ----------
    list_source_ind2D_weight_all: <list>
        Length of the list: number of meas cells (n_target)
        Each element: a list of ((2D-index), weight) of source cells
        that overlap with ONE target cell
    
    Requires
    ----------
    import xESMF as xe
    find_source_ind2D_weight
    '''
    
    # --- Load weight array in xESMF format --- #
    n_source = len(da_vic_domain['lat']) * len(da_vic_domain['lon'])
    n_target = len(da_meas['lat']) * len(da_meas['lon'])
    A = xe.frontend.read_weights(weight_nc, n_source, n_target)
    weight_array = A.toarray()  # [n_target, n_source]
    
    # --- For each measurement grid cell, find 2D-index & weight of all contributing VIC cells
    # each element: a list of ((2D-index), weight) of source cells
    #               that overlap with ONE target cell
    list_source_ind2D_weight_all = []
    # Loop over each target (measurement) grid cell
    for i in range(n_target):
        # Find corresponding VIC cell
        list_source_ind2D_weight = find_source_ind2D_weight(
            ind_target_1D=i, weight_array=weight_array,
            len_x_source=len(da_vic_domain['lon']))
        list_source_ind2D_weight_all.append(list_source_ind2D_weight)
    
    return list_source_ind2D_weight_all


def calculate_gain_K_whole_field_mismatched_grid(da_x, da_y_est, R,
                                                 list_source_ind2D_weight_all,
                                                 weight_nc, da_meas):
    ''' This function calculates gain K over the whole field, when the measurement
    grid mismatches the VIC grid.
    
    Parameters
    ----------
    da_x: <xr.DataArray>
        Soil moisture states of all ensemble members;
        As returned from get_soil_moisture_and_estimated_meas_all_ensemble;
        Dimension: [lat, lon, n, N]
    da_y_est: <xr.DataArray>
        Estimated measurement of all ensemble members;
        As returned from get_soil_moisture_and_estimated_meas_all_ensemble;
        Dimension: [lat, lon, m, N]
    R: <np.array> [m*m]
        Measurement error covariance matrix
    list_source_ind2D_weight_all: <list>
        Typically returned by extract_mismatched_grid_weight_info().
        list_source_ind2D_weight_all: <list>
        Length of the list: number of meas cells (n_target)
        Each element: a list of ((2D-index), weight) of source cells
        that overlap with ONE target cell
    weight_nc: <str>
        Weight file, pre-calculated by xESMF and process_weight_file()
    da_meas: <xr.DataArray>
        Measurement data. Only used for extracting lat lon info.
    
    Returns
    ----------
    list_K: <list>
        A list of K for each meas grid cell
        Each element of the list is a K matrix of dim [n_stacked, m]
        (n_stacked can be different for each meas cell)
    y_est_remapped: <np.array>
        y_est remapped to the meas grid.
        Dim: [n_target, m, N]
    '''
    
    # --- Extract some dimension info --- #
    # Total number of meas grid cells
    n_target = len(list_source_ind2D_weight_all)
    # Ensemble size N
    N = len(da_x['N'])
    
    # --- For each meas grid cell, --- #
    # --- stack states for all contributing source cells together (x) --- #
    # Notes: the first loop is over each meas grid cell;
    # the second loop is over each contributing VIC cell to this meas cell;
    # (item[0][0], item[0][1]) is the 2D-index of a source cell;
    # Original da_x: [lat, lon, n, N]
    # Each x_stacked in the list: [n_contrib_source_cells, n, N], then stack the first
    # two dimensions to be [n_contrib_source_cells*n, N]
    # Length of list: n_target
    list_x_stacked = [
        np.asarray(
            [da_x.values[item[0][0], item[0][1], :, :]
             for item in list_source_ind2D_weight_all[i]]).reshape([-1, N])
        for i in range(n_target)]
    
    # --- Aggregate y_est to meas grid for the whole domain --- #
    # remap_con needs lat lon to be the last two dims
    da_y_est_rolled = da_y_est.transpose('m', 'N', 'lat', 'lon')
    # Remap y_est
    da_y_est_remapped_rolled, weight_array = remap_con(
        reuse_weight=True, da_source=da_y_est_rolled,
        final_weight_nc=weight_nc, da_target_domain=da_meas)
    # Roll the y_est_remapped dimension back to original order
    da_y_est_remapped = da_y_est_remapped_rolled.transpose('lat', 'lon', 'm', 'N')
    # Flatten y_est_remapped into 1D index
    y_est_remapped = da_y_est_remapped.values.reshape(
        [-1, len(da_y_est_remapped['m']), len(da_y_est_remapped['N'])])  # [n_target, m, N]
    
    # --- Calculate gain K for each meas grid cell --- #
    # A list of K for each meas grid cell
    # Each element of the list is a K matrix of dim [n_stacked, m]
    # (n_stacked can be different for each meas cell)
    list_K = [calculate_gain_K(x=list_x_stacked[i], y_est=y_est_remapped[i, :, :], R=R)
              for i in range(n_target)]
    
    return list_K, y_est_remapped


def update_states_mismatched_grid(y_est_remapped, list_K, da_meas, R, state_nc_before_update,
                                  out_vic_state_nc, da_max_moist_n, list_source_ind2D_weight_all,
                                  adjust_negative=True, seed=None):
    ''' This function updates soil moisture states over the whole field,
    when the measurement grid mismatches the VIC grid.
    
    Parameters
    ----------
    y_est_remapped: <xr.DataArray>
        Estimated measurement from pre-updated states (y = Hx);
        Dimension: [n_target, m], where n_target is the number of meas grid cells
    list_K: <xr.DataArray>
        Gain K for the whole field
        Output from calculate_gain_K_whole_field_mismatched_grid()
    da_meas: <xr.DataArray> [lat, lon, m]
        Measurements at current time
    R: <float> (for m = 1)
        Measurement error covariance matrix (measurement error ~ N(0, R))
    state_nc_before_update: <str>
        VIC state file before update
    out_vic_state_nc: <str>
        Path for saving updated state file in VIC format;
    da_max_moist_n: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each tile
        [unit: mm]. Soil moistures above maximum after perturbation will
        be reset to maximum value.
        Dimension: [lat, lon, n]
    list_source_ind2D_weight_all: <list>
        Typically returned by extract_mismatched_grid_weight_info().
        list_source_ind2D_weight_all: <list>
        Length of the list: number of meas cells (n_target)
        Each element: a list of ((2D-index), weight) of source cells
        that overlap with ONE target cell
    adjust_negative: <bool>
        Whether or not to adjust negative soil moistures after update
        to zero.
        Default: True (adjust negative to zero)
    seed: <int or None>
        Seed for random number generator; this seed will only be used locally
        in this function and will not affect the upper-level code.
        None for not re-assign seed in this function, but using the global seed)
        Default: None
    '''
    
    # --- Extract some dimension info --- #
    m = len(da_meas['m'])
    n_target = len(da_meas['lat']) * len(da_meas['lon'])
    n_source = len(da_max_moist_n['lat']) * len(da_max_moist_n['lon'])
    n = len(da_max_moist_n['n'])
    
    # --- For each meas cell, calculate delta = K * (y_meas + v - y_est) --- #
    # Generate random measurement perturbation
    if seed is None:
        v = np.random.multivariate_normal(
                np.zeros(m), R, size=n_target).reshape([n_target, m, 1])  # [n_target, m, 1]
    else:
        rng = np.random.RandomState(seed)
        v = rng.multivariate_normal(
                np.zeros(m), R, size=n_target).reshape([n_target, m, 1])  # [n_target, m, 1]
    # Flatten the measurement data to 1D
    meas_time_flat = da_meas.values.reshape(
        [len(da_meas['lat'])*len(da_meas['lon']), m, 1])  # [n_target, m, 1]
    # Calculate delta for the whole domain
    # If there is some contributing cells, and meas for this cell is not NAN;
    # Otherwise, update delta is zero (i.e., no update)
    # Length of list_delta: n_target
    # Each member of list_delta is of dim [n_stacked, 1]
    list_delta = [np.dot(list_K[i],
                         meas_time_flat[i, :, :] + v[i, :, :] - y_est_remapped[i, :])
                  if (list_K[i].shape[0] > 0 and ~np.isnan(meas_time_flat[i].squeeze()))
                  else np.ones([list_K[i].shape[0], 1]) * 99999
                  for i in range(n_target)]
    
    # --- Re-distribute delta to original VIC cells for the whole domain --- #
    # Initialize the VIC-cell-level delta. Dim: [n_source, n]
    delta_vic_flat = np.zeros([n_source, n])
    # Add up the actual delta according to weight
    for i_target in range(n_target):
        # Identify number of contributing cells
        n_contrib = len(list_source_ind2D_weight_all[i_target])
        # Unstack delta for each contributing VIC cell
        delta_unstacked = list_delta[i_target].reshape([n_contrib, n])
        # Loop over each contributing cell
        for j in range(n_contrib):
            # Identify contributing cell index and weight
            ind_source_2D = list_source_ind2D_weight_all[i_target][j][0]
            weight = list_source_ind2D_weight_all[i_target][j][1]
            # Weighted-add unstacked delta to the final delta
            ind_source_1D = map_ind_2D_to_1D(ind_2D_lat=ind_source_2D[0],
                                             ind_2D_lon=ind_source_2D[1],
                                             len_x=len(da_max_moist_n['lon']))
            delta_vic_flat[ind_source_1D, :] += delta_unstacked[j, :] * weight
    # Reshape delta back to 2D VIC domain [lat, lon, n]
    delta = delta_vic_flat.reshape([len(da_max_moist_n['lat']),
                                        len(da_max_moist_n['lon']), n])
    
    # --- Add delta to original VIC states --- #
    # Load before-update state file
    ds = xr.open_dataset(state_nc_before_update)
    class_states = States(ds)
    # Initiate DataArray for updated states
    da_x_updated = class_states.da_EnKF.copy(deep=True)  # [lat, lon, n]
    # Update states - add delta to orig. states
    da_x_updated[:] += delta
    
    # --- Reset negative updated soil moistures to zero --- #
    x_new = da_x_updated[:].values
    if adjust_negative:
        x_new[x_new<0] = 0

    # --- Reset updated soil moistures above maximum to maximum --- #
    max_moist = da_max_moist_n.values
    x_new[(x_new>max_moist)] = max_moist[(x_new>max_moist)]

    # --- Put into da --- #
    da_x_updated[:] = x_new
    
    # --- Save some diagnostic variables --- #
    da_update_increm = da_x_updated - class_states.da_EnKF
    
    # --- Save updated states to netCDF --- #
    # Convert updated states to VIC states format
    ds_updated = class_states.convert_new_EnKFstates_sm_to_VICstates(da_x_updated)
    to_netcdf_state_file_compress(ds_updated, out_vic_state_nc)
    
    return da_x_updated, da_update_increm


def find_source_ind2D_weight(ind_target_1D, weight_array, len_x_source):
    ''' Given a weight array in xESMF foramt ([n_target, n_source]), and the
    1D index of ONE target grid cell, return the 2D index of all the corresponding
    source grid cells.
    
    Parameters
    ----------
    ind_target_1D: <int>
        Index of a target grid cell in the 1D flattened array. Index starts from 0.
    weight_array: <np.array>
        Weight array in xESMF format. Dimension: [n_target, n_source]
    len_x_source: <int>
        Length of x (lon) in the 2D domain of source.
    
    Returns
    ----------
    list_ind2D_weight_source: <list>
        A list of tuples of ((ind_lat, ind_lon), weight) of the source domain.
        The length of the list is the totol number of source cells that contribute
        weights to this target cell. Each tuple is the 2D index (starting from 0) of
        each contributing source cell.
    
    Requires
    ----------
    map_coord_1D_to_2D
    '''
    
    list_ind2D_weight_source = [(map_ind_1D_to_2D(ind, len_x_source), weight_array[ind_target_1D, ind])
                         for ind in np.where(weight_array[ind_target_1D, :] > 0)[0]]
    
    return list_ind2D_weight_source


def map_ind_1D_to_2D(ind_1D, len_x):
    ''' Maps an index of a flattened array to an index in a 2D domain.
        The 2D domain [lat, lon] is flattened into 1D as 2D.reshape([lat*lon]).
    
    Parameters
    ----------
    ind_1D: <int>
        Index in a flattened array. Index starts from 0.
    len_x: <int>
        Length of x (lon) in the 2D domain. Index starts from 0.
    
    Returns
    ----------
    ind_2D: (<int>, <int>)
        Index in the 2D array. Index starts from 0. (lat, lon) order.
    '''
    
    ind_2D_lat = ind_1D // len_x
    ind_2D_lon = ind_1D % len_x
    
    return(ind_2D_lat, ind_2D_lon)


def map_ind_2D_to_1D(ind_2D_lat, ind_2D_lon, len_x):
    ''' Maps an index of a 2D array to an index in a flattened 1D array.
        The 2D domain [lat, lon] is flattened into 1D as 2D.reshape([lat*lon]).
    
    Parameters
    ----------
    ind_2D_lat: <int>
        y or lat index in the 2D array. Index starts from 0.
    ind_2D_lon: <int>
        y or lon index in the 2D array. Index starts from 0.
    len_x: <int>
        Length of x (lon) in the 2D domain. Index starts from 0.
    
    Returns
    ----------
    ind_1D: <int>
        Index in the flattened array. Index starts from 0.
    '''
    
    if ind_2D_lon < len_x:    
        ind_1D = ind_2D_lat * len_x + ind_2D_lon
    else:
        raise ValueError('x or lon index in a 2D domain exceeds the dimension!')
    
    return(ind_1D)


def remap_con(reuse_weight, da_source, final_weight_nc, da_target_domain,
              da_source_domain=None,
              tmp_weight_nc=None, process_method=None):
    ''' Conservative remapping

    Parameters
    ----------
    reuse_weight: <bool>
        Whether to use an existing weight file directly, or to calculate weights
    da_source: <xr.DataArray>
        Source data. The dimension names must be "lat" and "lon".
    final_weight_nc: <str>
        If reuse_weight = False, path for outputing the final weight file;
        if reuse_weight = True, path for the weight file to use for regridding
    da_target_domain: <xr.DataArray>
        Domain of the target grid.
    da_source_domain: <xr.DataArray> (Only needed when reuse_weight = False)
        Domain of the source grid.
    tmp_weight_nc: <str> (Only needed when reuse_weight = False)
        Path for outputing the temporary weight file from xESMF
    process_method: (Only needed when reuse_weight = False)
        This option is not implemented yet (right now, there is only one way of processing
        the weight file). Can be extended to have the option of, e.g., setting a threshold
        for whether to remap for a target cell or not if the coverage is low

    Return
    ----------
    da_remapped: <xr.DataArray>
        Remapped data

    Requires
    ----------
    process_weight_file
    import xesmf as xe
    '''

    # --- Grab coordinate information --- #
    src_lons = da_source['lon'].values
    src_lats = da_source['lat'].values
    target_lons = da_target_domain['lon'].values
    target_lats = da_target_domain['lat'].values
    lon_in_edges = edges_from_centers(src_lons)
    lat_in_edges = edges_from_centers(src_lats)
    lon_out_edges = edges_from_centers(target_lons)
    lat_out_edges = edges_from_centers(target_lats)
    grid_in = {'lon': src_lons,
               'lat': src_lats,
               'lon_b': lon_in_edges,
               'lat_b': lat_in_edges}
    grid_out = {'lon': target_lons,
                'lat': target_lats,
                'lon_b': lon_out_edges,
                'lat_b': lat_out_edges}
    # --- If reuse_weight = False, calculate weights --- #
    if reuse_weight is False:
        # Create regridder using xESMF
        regridder = xe.Regridder(grid_in, grid_out, 'conservative',
                                 filename=tmp_weight_nc)
        # Process the weight file to be correct, and save to final_weight_nc
        weight_array = process_weight_file(
            tmp_weight_nc, final_weight_nc,
            len(src_lons) * len(src_lats),
            len(target_lons) * len(target_lats),
            da_source_domain,
            process_method=None)  # weight_array: [n_target, n_source]
    # --- Use the final weight file to regrid input data --- #
    regridder = xe.Regridder(grid_in, grid_out, 'conservative',
                             filename=final_weight_nc, reuse_weights=True)
    weight_array = regridder.A.toarray()  # extract weight_array
    da_remapped = regridder(da_source)
    # If weight for a target cell is negative, it means that the target cell
    # does not overlap with any valid source cell. Thus set the remapped value to NAN
    nan_weights = (weight_array.sum(axis=1).reshape([len(target_lats), len(target_lons)]) < 0)
    data = da_remapped.values
    data[..., nan_weights] = np.nan
    da_remapped[:] = data

    return da_remapped, weight_array


def edges_from_centers(centers):
    ''' Return an array of grid edge values from grid center values
    Parameters
    ----------
    centers: <np.array>
        A 1-D array of grid centers. Typically grid-center lats or lons. Dim: [n]

    Returns
    ----------
    edges: <np.array>
        A 1-D array of grid edge values. Dim: [n+1]
    '''

    edges = np.zeros(len(centers)+1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])

    return edges




