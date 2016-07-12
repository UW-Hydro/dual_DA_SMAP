
import numpy as np
import pandas as pd
import os
import string
from collections import OrderedDict
import xarray as xr
import multiprocessing as mp

from tonic.models.vic.vic import VIC, default_vic_valgrind_error_code


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
                where len(n) = len(veg_class) * len(snow_band) * len(nlayer)
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
        EnKF_states = self.ds['STATE_SOIL_MOISTURE'].values.reshape(
                                                [n, len(lat), len(lon)])
        
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
        ds = self.ds.copy()
        
        # Convert da_EnKF dimension
        EnKF_states_reshape = da_EnKF.values.reshape(len(lat), len(lon), len(veg_class),
                                                 len(snow_band), len(nlayer))
        EnKF_states_reshape = np.rollaxis(EnKF_states_reshape, 0, 5)  # put lat and lon to the last
        EnKF_states_reshape = np.rollaxis(EnKF_states_reshape, 0, 5)
        
        # Fill into VIC states ds
        ds['STATE_SOIL_MOISTURE'][:] = EnKF_states_reshape
        
        return ds
        
    
    def add_gaussian_white_noise_states(self, P):
        ''' Add a constant Gaussian noise (constant covariance matrix of all states n
            over all grid cells) for the whole field.
            NOTE: this method does not change self

        Parameters
        ----------
        P: <float>
            Covariance matrix of the Gaussian noise to be added
            
        Returns
        ----------
        ds: <xr.dataset>
            A dataset of VIC states, with soil moisture states = da_EnKF, and all the other
            state variables = those in self.ds
        '''

        # Extract the number of EnKF states
        n = len(self.da_EnKF['n'])
        
        # Check if P is in shape [n*n]
        pass
        
        # Generate random noise for the whole field
        noise = np.random.multivariate_normal(
                        np.zeros(n), P,
                        size=len(self.ds['lat'])*\
                             len(self.ds['lon']))
        noise_reshape = noise.reshape([n,
                                       len(self.ds['lat']),
                                       len(self.ds['lon'])])
        noise_reshape = np.rollaxis(noise_reshape, 0, 3)  # roll the 'n' dimension to the last
        
        # Add noise to soil moisture field
        da_perturbed = self.da_EnKF.copy()
        da_perturbed[:] += noise_reshape

        # Reset negative perturbed soil moistures to zero
        sm_new = da_perturbed.values
        sm_new[sm_new<0] = 0
        da_perturbed[:] = sm_new
        
        # Put the perturbed soil moisture states to back to VIC states ds
        ds = self.convert_new_EnKFstates_sm_to_VICstates(da_perturbed)

        return ds
    
    
    def perturb_soil_moisture_Gaussian(self, global_path, sigma_percent):
        ''' Perturb each soil moisture state by sigma_percent %, uncorrelated
            Gaussian noise.
        
        Parameters
        ----------
        global_path: <str>
            VIC global parameter file path; can be a template file (here it is only used to
            extract soil parameter file info)
        sigma_percent: <float>
            Percentage of the maximum state value to perturb; sigma_percent will be used
            as the standard deviation of the Gaussian noise added (e.g., sigma_percent = 5
            for 5% of maximum soil moisture perturbation)
        
        Returns
        ----------
        ds: <xr.dataset>
            A dataset of VIC states, with soil moisture states perturbed, and all the other
            state variables = those in self.ds
        
        Require
        ----------
        numpy
        find_global_param_value
        '''
        
        # Load soil parameter file (as defined in global file)
        with open(global_path, 'r') as global_file:
            global_param = global_file.read()
        soil_nc = find_global_param_value(global_param, 'SOIL')
        ds_soil = xr.open_dataset(soil_nc, decode_cf=False)
        ds_soil.load()
        ds_soil.close()
        
        # Calculate maximum soil moisture for each layer
        # Dimension: [nlayer, lat, lon]
        da_depth = ds_soil['depth']  # [m]
        da_bulk_density = ds_soil['bulk_density']  # [kg/m3]
        da_soil_density = ds_soil['soil_density']  # [kg/m3]
        da_porosity = 1 - da_bulk_density / da_soil_density
        da_max_moist = da_depth * da_porosity * 1000  # [mm]

        # Perturb for each layer and each grid cell
        veg_class = self.ds['veg_class']
        snow_band = self.ds['snow_band']
        lat = self.ds['lat']
        lon = self.ds['lon']
        nlayer = self.ds['nlayer']
        
        ds = self.ds.copy()
        for lt in lat:
            for lg in lon:
                for l in nlayer:
                    max_moist = da_max_moist.loc[l, lt, lg].values
                    if np.isnan(max_moist) == True or max_moist <= 0:  # if inactive cell
                        continue
                    noise = np.random.normal(
                                    loc=0,
                                    scale=da_max_moist.loc[l, lt, lg].values *\
                                          sigma_percent / 100.0,
                                    size=(len(veg_class), len(snow_band)))
                    ds['STATE_SOIL_MOISTURE'].loc[:, :, l, lt, lg] += noise
        
        # Reset negative perturbed soil moistures to zero
        sm_new = ds['STATE_SOIL_MOISTURE'].values
        sm_new[sm_new<0] = 0
        ds['STATE_SOIL_MOISTURE'][:] = sm_new

        return ds
        
    
    def update_soil_moisture(self, da_K, da_y, da_y_est, R):
        ''' This function updates soil moisture states for the whole field
        
        Parameters
        ----------
        da_K: <xr.DataArray>
            Gain K for the whole field;
            Dimension: [veg_class, snow_band, lat, lon, n, m], 
            where [n, m] is the Kalman gain K;
            This is output from function calculate_gain_K_whole_field
        da_y: <xr.DataArray>
            Measurement at this time point for the whole field;
            Dimension: [lat, lon]
            (measurement for each grid cell )
        '''
        pass


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
        
    
    def perturb_prec_lognormal(self, varname, std=1):
        ''' Perturb precipitation forcing data
        
        Parameters
        ----------
        varname: <str>
            Precipitation variable name in the 
        std: <float>
            Standard deviation of the multiplier (which is log-normal distributed);
            Note: here sigma is the real standard deviation of the multiplier, not
            the standard deviation of the underlying normal distribution! (and the
            real mean of the multiplier is 1)
        '''
        
        # Calculate mu and sigma for the lognormal distribution
        mu = -0.5 * np.log(std+1)
        sigma = np.sqrt(np.log(std+1))
        
        # Generate random noise for the whole field
        noise = np.random.lognormal(mu, sigma,
                                    size=(self.time_len, self.lat_len, self.lon_len))
        
        # Add noise to soil moisture field
        ds_perturbed = self.ds.copy()
        ds_perturbed[varname][:] *= noise
        
        return ds_perturbed


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

    def add_gaussian_white_noise(self, da_sigma):
        ''' Add Gaussian noise for all active grid cells

        Parameters
        ----------
        sigma: <xarray.DataArray>
            Standard deviation of the Gaussian white noise to add, can be spatially different
            for each grid cell (but temporally constant);
            Dimension: [lat, lon]
        
        Returns
        ----------
        da_perturbed: <xarray.DataArray>
            Perturbed variable for the whole field
            Dimension: [time, lat, lon]
        '''

        # Generate random noise for the whole field
        da_noise = self.da.copy()
        da_noise[:] = np.nan
        for lt in self.lat:
            for lg in self.lon:
                sigma = da_sigma.loc[lt, lg].values
                if np.isnan(sigma) == True or sigma <= 0:  # if inactive cell, skip
                    continue
                da_noise.loc[:, lt, lg] = np.random.normal(loc=0, scale=sigma, size=len(self.time))
        # Add noise to the original da
        da_perturbed = self.da + da_noise
        # Set negative to zero
        tmp = da_perturbed.values
        tmp[tmp<0] = 0
        da_perturbed[:] = tmp
        # Add attrs back
        da_perturbed.attrs = self.da.attrs

        return da_perturbed


def EnKF_VIC(N, start_time, end_time, init_state_basepath, P0, R, da_meas,
             da_meas_time_var, vic_exe, vic_global_template,
             vic_forcing_orig_basepath,
             vic_model_steps_per_day, output_vic_global_root_dir,
             output_vic_state_root_dir, output_vic_history_root_dir,
             output_vic_forcing_root_dir, output_vic_log_root_dir,
             dict_varnames, prec_std, state_perturb_sigma_percent):
    ''' This function runs ensemble kalman filter (EnKF) on VIC (image driver)

    Parameters
    ----------
    N: <int>
        Number of ensemble members
    start_time: <pandas.tslib.Timestamp>
        Start time of EnKF run
    end_time: <pandas.tslib.Timestamp>
        End time of EnKF run
    init_state_basepath: <str>
        Initial state directory and file name prefix (excluding '.YYMMDD_SSSSS.nc');
        Initial state time must be one time step before start_time
    P0: <float>
        Initial state error matrix
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
    vic_forcing_orig_basepath: <str>
        VIC original unperturbed forcing netCDF file basepath ('YYYY.nc' will be appended)
    vic_model_steps_per_day: <int>
        VIC model steps per day
    output_vic_global_root_dir: <str>
        Directory for VIC global files
    output_vic_state_root_dir: <str>
        Directory for VIC output state files
    output_vic_result_root_dir: <str>
        Directory for VIC output result files
    output_vic_forcing_root_dir: <str>
        Directory for VIC perturbed forcing files
    output_vic_log_root_dir: <str>
        Directory for VIC output log files
    dict_varnames: <dict>
        A dictionary of forcing names in nc file;
        e.g., {'PREC': 'prcp'; 'AIR_TEMP': 'tas'}
    prec_std: <float>
        Standard deviation of the precipitation perturbing multiplier
    state_perturb_sigma_percent: <float>
        Percentage of max value of each state to perturb (e.g., if
        state_perturb_sigma_percent = 5, then Gaussian noise with standard deviation
        = 5% of max soil moisture will be added as perturbation)

    Returns
    ----------
    dict_ens_list_history_files: <dict>
        A dictory of lists;
        Keys: 'ens<i>', where i = 1, 2, ..., N
        Items: a list of output history files (in order) for this ensemble member

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
    da_tile_frac = determine_tile_frac(vic_global_template)
    
    # Check whether the dimension of P0 is consistent with number of soil moisture states
    pass
    # Check whether the run period is consistent with VIC setup
    pass
    # Check whether the time range of da_meas is within run period
    pass
    
    # --- Prepare perturbed forcing data for each ensemble member --- #
    print('\tPreparing perturbed forcings...')
    start_year = start_time.year
    end_year = end_time.year
    for year in range(start_year, end_year+1): 
        class_forcings_orig = Forcings(xr.open_dataset(
                                    '{}{}.nc'.format(vic_forcing_orig_basepath,
                                                     year)))
        perturb_forcings_ensemble(N, orig_forcing=class_forcings_orig,
                                  year=year, dict_varnames=dict_varnames,
                                  prec_std=prec_std,
                                  out_forcing_dir=output_vic_forcing_root_dir)
    
    # --- Step 1. Initialize ---#
    init_state_time = start_time - pd.DateOffset(hours=24/vic_model_steps_per_day)
    print('\tGenerating ensemble initial states at ', init_state_time)
    # Load initial state file
    ds_states = xr.open_dataset('{}.{}_{:05d}.nc'.format(
                                        init_state_basepath,
                                        init_state_time.strftime('%Y%m%d'),
                                        init_state_time.hour*3600+init_state_time.second))
    class_states = States(ds_states)
    
    # Determine the number of EnKF states, n
    n = len(class_states.da_EnKF['n'])
    # Set up initial state subdirectories
    init_state_dir_name = 'init.{}_{:05d}'.format(
                                    init_state_time.strftime('%Y%m%d'),
                                    init_state_time.hour*3600+init_state_time.second)
    init_state_dir = setup_output_dirs(
                            output_vic_state_root_dir,
                            mkdirs=[init_state_dir_name])[init_state_dir_name]
    # For each ensemble member, add Gaussian noise to sm states with covariance P0,
    # and save each ensemble member states
    P0_diag = np.identity(n) * P0  # Set up P0 matrix
    for i in range(N):
        ds = class_states.add_gaussian_white_noise_states(P0_diag)
        ds.to_netcdf(os.path.join(init_state_dir,
                                  'state.ens{}.nc'.format(i+1)),
                     format='NETCDF4_CLASSIC')
    
    # --- Step 2. Propagate (run VIC) until the first measurement time point ---#    
    # Initialize dictionary of history file paths for each ensemble member
    dict_ens_list_history_files = {}
    for i in range(N):
        dict_ens_list_history_files['ens{}'.format(i+1)] = []

    # Determine VIC run period
    vic_run_start_time = start_time
    vic_run_end_time = pd.to_datetime(da_meas[da_meas_time_var].values[0])
    print('\tPropagating (run VIC) until the first measurement time point ',
          vic_run_end_time)
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
    propagate_ensemble(N, start_time=vic_run_start_time, end_time=vic_run_end_time,
                       vic_exe=vic_exe,
                       vic_global_template_file=vic_global_template,
                       vic_model_steps_per_day=vic_model_steps_per_day,
                       init_state_dir=init_state_dir,
                       out_state_dir=out_state_dir,
                       out_history_dir=out_history_dir,
                       out_global_dir=out_global_dir,
                       out_log_dir=out_log_dir,
                       forcing_perturbed_dir=output_vic_forcing_root_dir)
    # Put output history file paths into dictionary
    for i in range(N):
        dict_ens_list_history_files['ens{}'.format(i+1)].append(os.path.join(
                    out_history_dir, 'history.ens{}.{}-{:05d}.nc'.format(
                            i+1,
                            vic_run_start_time.strftime('%Y-%m-%d'),
                            vic_run_start_time.hour*3600+vic_run_start_time.second)))
    
    # --- Step 3. Run EnKF --- #
    # Initialize
    state_dir_before_update = out_state_dir
    
    # Loop over each measurement time point
    for t, time in enumerate(da_meas[da_meas_time_var]):

        # Determine last, current and next measurement time points
        last_time = pd.to_datetime(time.values)
        current_time = last_time + pd.DateOffset(hours=24/vic_model_steps_per_day)
        if t == len(da_meas[da_meas_time_var])-1:  # if this is the last measurement time
            next_time = end_time
        else:  # if not the last measurement time
            next_time = pd.to_datetime(da_meas[da_meas_time_var][t+1].values)
        print('\tCalculating for ', current_time, 'to', next_time)
        
        # (1) Calculate gain K
        da_x, da_y_est = get_soil_moisture_and_estimated_meas_all_ensemble(
                                N,
                                state_dir=state_dir_before_update,
                                state_time=last_time, 
                                da_tile_frac=da_tile_frac)
        da_K = calculate_gain_K_whole_field(da_x, da_y_est, R)

        # (2) Update states for each ensemble member
        # Set up dir for updated states
        updated_states_dir_name = 'updated.{}_{:05d}'.format(
                                        last_time.strftime('%Y%m%d'),
                                        last_time.hour*3600+last_time.second)
        out_updated_state_dir = setup_output_dirs(output_vic_state_root_dir,
                                                  mkdirs=[updated_states_dir_name])[updated_states_dir_name]
        # Update states and save to nc files
        da_x_updated = update_states_ensemble(da_x, da_y_est, da_K,
                                               da_meas.loc[time, :, :, :], R,
                                               state_dir_before_update=state_dir_before_update,
                                               current_time=last_time,
                                               out_vic_state_dir=out_updated_state_dir)
        
        # (3) Propagate each ensemble member to the next measurement time point
        # If current_time > next_time, do not propagate (we already reach the end of the simulation)
        if current_time > next_time:
            break
        # --- Perturb states --- #
        # Set up perturbed state subdirectories
        pert_state_dir_name = 'perturbed.{}_{:05d}'.format(
                                        last_time.strftime('%Y%m%d'),
                                        last_time.hour*3600+last_time.second)
        pert_state_dir = setup_output_dirs(
                            output_vic_state_root_dir,
                            mkdirs=[pert_state_dir_name])[pert_state_dir_name]
        
        # Perturb states for each ensemble member
        perturb_soil_moisture_states_ensemble(
                    N,
                    states_to_perturb_dir=out_updated_state_dir,
                    global_path=vic_global_template,
                    sigma_percent=state_perturb_sigma_percent,
                    out_states_dir=pert_state_dir)
        
        # --- Propagate to the next time point --- #
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
        propagate_ensemble(N, start_time=current_time, end_time=next_time,
                       vic_exe=vic_exe,
                       vic_global_template_file=vic_global_template,
                       vic_model_steps_per_day=vic_model_steps_per_day,
                       init_state_dir=pert_state_dir,  # perturbes states as init state
                       out_state_dir=out_state_dir,
                       out_history_dir=out_history_dir,
                       out_global_dir=out_global_dir,
                       out_log_dir=out_log_dir,
                       forcing_perturbed_dir=output_vic_forcing_root_dir)  # perturbed forcing

        # Put output history file paths into dictionary
        for i in range(N):
            dict_ens_list_history_files['ens{}'.format(i+1)].append(os.path.join(
                    out_history_dir, 'history.ens{}.{}-{:05d}.nc'.format(
                            i+1,
                            current_time.strftime('%Y-%m-%d'),
                            current_time.hour*3600+current_time.second)))
        
        # Point state directory to be updated to the propagated one
        state_dir_before_update = out_state_dir

    return dict_ens_list_history_files


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
        Model run end time
    init_state: <str>
        A full line of initial state option in the global file.
        E.g., "# INIT_STATE"  for no initial state;
              or "INIT_STATE /path/filename" for an initial state file
    vic_state_basepath: <str>
        Output state name directory and file name prefix
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
    '''
    
    # --- Create template string --- #
    with open(global_template_path, 'r') as global_file:
        global_param = global_file.read()

    s = string.Template(global_param)
    
    # --- Fill in global parameter options --- #
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
                                     stateyear=end_time.year, # save state at end_time
                                     statemonth=end_time.month,
                                     stateday=end_time.day,
                                     statesec=end_time.hour*3600+end_time.second,
                                     result_dir=vic_history_file_dir)
    
    # --- Replace global parameters in replace --- #
    global_param = replace_global_values(global_param, replace)
    
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


def run_vic_for_multiprocess(vic_exe, global_file, log_dir):
    '''This function is a simple wrapper for calling "run" method under
        VIC class in multiprocessing

    Parameters
    ----------
    vic_exe: <class VIC>
        A VIC class object
    global_file: <str>
        VIC global file path
    log_dir: <str>
        VIC run output log directory

    Require
    ----------
    check_returncode
    '''

    returncode = vic_exe.run(global_file, logdir=log_dir)
    check_returncode(returncode, expected=0)


def propagate_ensemble(N, start_time, end_time, vic_exe, vic_global_template_file,
                       vic_model_steps_per_day, init_state_dir, out_state_dir,
                       out_history_dir, out_global_dir, out_log_dir,
                       forcing_perturbed_dir):
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
    forcing_perturbed_dir: <str>
        Perturbed forcing directory. File names are: "forc.ens<i>.<YYYY>.nc",
        where <i> is 1, 2, ..., N, and <YYYY> is forcing year
        
    Require
    ----------
    OrderedDict
    multiprocessing
    generate_VIC_global_file
    '''
    
    # --- Set up multiprocessing --- #
    pool = mp.Pool(processes=20)
    
    # --- Loop over each ensemble member --- #
    for i in range(N):
        # Generate VIC global param file
        replace = OrderedDict([('FORCING1', os.path.join(forcing_perturbed_dir,
                                                         'forc.ens{}.'.format(i+1))),
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
        pool.apply_async(run_vic_for_multiprocess, (vic_exe, global_file, out_log_dir,))
        #returncode = vic_exe.run(global_file, logdir=out_log_dir)
        #check_returncode(returncode, expected=0)
    
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
    
    # Load global parameter file
    with open(global_path, 'r') as global_file:
            global_param = global_file.read()
            
    # Extract Cv from vegparam file (as defined in the global file)
    vegparam_nc = find_global_param_value(global_param, 'VEGPARAM')   
    da_Cv = xr.open_dataset(vegparam_nc, decode_cf=False)['Cv']  # dim: [veg_class, lat, lon]
    lat = da_Cv['lat']
    lon = da_Cv['lon']
    
    # Extract snowband info from the global file
    # Dimension of da_AreaFract: [snowband, lat, lon]
    n_snowband = int(find_global_param_value(global_param, 'SNOW_BAND'))
    if n_snowband == 1:  # if only one snowband
        data = np.ones([1, len(lat), len(lon)])
        da_AreaFract = xr.DataArray(data, coords=[[0], lat, lon],
                                    dims=['snow_band', 'lat', 'lon'])
    else:  # if more than one snowband
        tmp, snowband_nc = find_global_param_value(global_param, 'SNOW_BAND',
                                                          second_param=True)
        da_AreaFract = xr.open_dataset(snowband_nc, decode_cf=False)['AreaFract']

    # Initialize the final DataArray
    veg_class = da_Cv['veg_class']
    snow_band = da_AreaFract['snow_band']
    data = np.empty([len(veg_class), len(snow_band), len(lat), len(lon)])
    data[:] = np.nan
    da_tile_frac = xr.DataArray(data, coords=[veg_class, snow_band, lat, lon],
                                dims=['veg_class', 'snow_band', 'lat', 'lon'])
    
    # Calculate fraction of each veg/snowband tile for each grid cell, and fill in
    # da_file_frac
    for lt in lat:
        for lg in lon:
            da_tile_frac.loc[:, :, lt, lg] =\
                    da_Cv.loc[:, lt, lg] * da_AreaFract.loc[:, lt, lg]
    
    return da_tile_frac


def get_soil_moisture_and_estimated_meas_all_ensemble(N, state_dir, state_time,
                                                      da_tile_frac):
    ''' This function extracts soil moisture states from netCDF state files for all ensemble
        members, for all grid cells, veg and snow band tiles.
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    state_dir: <str>
        Directory of state files for each ensemble member
        State file names are "state.ens<i>.xxx.nc", where <i> is 1, 2, ..., N
    state_time: <pandas.tslib.Timestamp>
        State time
    da_tile_frac: <xr.DataArray>
        Fraction of each veg/snowband in each grid cell for the whole domain
        Dimension: [veg_class, snow_band, lat, lon]

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
    state_name = 'state.ens1.{}_{}.nc'.format(state_time.strftime('%Y%m%d'),
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
    for i in range(N):
        # --- Load state file --- #
        state_name = 'state.ens{}.{}_{}.nc'.format(i+1,
                                                   state_time.strftime('%Y%m%d'),
                                                   state_time.hour*3600+state_time.second)
        ds = xr.open_dataset(os.path.join(state_dir, state_name))
        class_states = States(ds)
        
        # --- Fill x and y_est data in --- #
        # Fill in states x
        da_x.loc[:, :, :, i] = class_states.da_EnKF
        # Fill in measurement estimates y
        da_y_est[:, :, :, i] = calculate_y_est_whole_field(
                                        class_states.ds['STATE_SOIL_MOISTURE'],
                                        da_tile_frac)
        
    return da_x, da_y_est


def calculate_y_est(da_x_cell, da_tile_frac_cell):
    ''' Caclulate estimated measurement y_est = Hx for one grid cell; here y_est is
        calculated as tile-average top-layer soil moisture over the whole grid cell.
    
    Parameters
    ----------
    da_x_cell: <xr.DataArray>
        An DataArray of VIC soil moisture states for a grid cell
        Dimension: [veg_class, snow_band, nlayer]
    da_tile_frac_cell: <xr.DataArray>
        An DataArray of veg/band tile fraction for a grid cell
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
    y_est = np.nansum(da_x_cell[:, :, 0].values * da_tile_frac_cell.values)
    
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
    
    # Extract lat and lon coords
    lat = da_x['lat']
    lon = da_x['lon']
    
    # Initiate da_y_est
    data = np.empty([len(lat), len(lon), 1])
    da_y_est = xr.DataArray(data, coords=[lat, lon, [0]], dims=['lat', 'lon', 'm'])
    
    # Loop over each grid cell and calculate y_est, and fill in da_y_est
    for lt in lat:
        for lg in lon:
            da_y_est.loc[lt, lg, 0] = calculate_y_est(da_x.loc[:, :, :, lt, lg],
                                                      da_tile_frac.loc[:, :, lt, lg])
    
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
    m_coord = da_y_est['m']
    
    # --- Initialize da_K --- #
    K = np.empty([len(lat_coord), len(lon_coord), len(n_coord), len(m_coord)])
    K[:] = np.nan
    da_K = xr.DataArray(K,
                        coords=[lat_coord, lon_coord, n_coord, m_coord],
                        dims=['lat', 'lon', 'n', 'm'])
    
    # --- Calculate gain K for the whole field --- #
    for lat in lat_coord:
        for lon in lon_coord:
            # Calculate gain K
            K = calculate_gain_K(da_x.loc[lat, lon, :, :],
                                 da_y_est.loc[lat, lon, :, :], R)
            # Fill K in da_K
            da_K.loc[lat, lon, :, :] = K
    
    return da_K


def update_states_ensemble(da_x, da_y_est, da_K, da_meas, R, state_dir_before_update, current_time,
                            out_vic_state_dir):
    ''' Update the EnKF states for the whole field for each ensemble member.
    
    Parameters
    ----------
    da_x: <xr.DataArray>
        Soil moisture states of all ensemble members, before update;
        Dimension: [lat, lon, n, N]
    da_y_est: <xr.DataArray>
        Estimated measurement from pre-updated states of all ensemble members (y = Hx);
        Dimension: [lat, lon, m, N]
    da_K: <xr.DataArray>
        Gain K for the whole field
        Dimension: [lat, lon, n, m], where [n, m] is the Kalman gain K
    da_meas: <xr.DataArray> [lat, lon, m]
        Measurements at current time
    R: <float> (for m = 1)
        Measurement error covariance matrix (measurement error ~ N(0, R))
    state_dir_before_update: <str>
        Directory of VIC states before update;
        State file names are: state.ens<i>.<YYYYMMDD>_<SSSSS>.nc,
        where <i> is ensemble member index (1, ..., N),
              <YYYYMMMDD>_<SSSSS> is the current time of the states
    current_time: <pandas.tslib.Timestamp>
        Current time of the states
    output_vic_state_dir: <str>
        Directory for saving updated state files in VIC format;
        State file names will be: state.ens<i>.nc, where <i> is ensemble member index (1, ..., N)
    
    Returns
    ----------
    da_x_updated: <xr.DataArray>
        Updated soil moisture states;
        Dimension: [lat, lon, n, N]
    
    Require
    ----------
    numpy
    '''
    
    # Extract dimensions
    N = len(da_x['N'])  # number of ensemble members
    m = len(da_y_est['m'])
    n = len(da_x['n'])
    lat_coord = da_x['lat']
    lon_coord = da_x['lon']
    
    # Initiate updated DataArray for states
    da_x_updated = da_x.copy()
    
    # Loop over each ensemble member and each grid cell
    for i in range(N):
        for lat in lat_coord:
            for lon in lon_coord:
                # --- Calculate delta = K * (y_meas + v - y_est) for all grid cells --- #
                # Generate random measurement perturbation
                v = np.random.multivariate_normal(np.zeros(m), R).reshape((m, 1))  # [m*1]
                # Extract other data for this grid cell
                K = da_K.loc[lat, lon, :, :].values  # [n*m]
                y_meas = da_meas.loc[lat, lon, :].values.reshape((m, 1))  # [m*1]
                y_est = da_y_est.loc[lat, lon, :, i].values.reshape((m, 1))  # [m*1]
                delta = np.dot(K, y_meas + v - y_est)  # [n*1]
                # --- Update states --- #
                da_x_updated.loc[lat, lon, :, i] += delta.reshape((n))
                # --- Set negative to zero --- #
                tmp = da_x_updated.loc[lat, lon, :, i].values
                tmp[tmp<0] = 0
                da_x_updated.loc[lat, lon, :, i] = tmp
                
        # --- Save updated states to nc files for each ensemble member --- #
        # Load VIC states before update for this ensemble member          
        ds = xr.open_dataset(os.path.join(state_dir_before_update,
                                          'state.ens{}.{}_{:05d}.nc'.format(
                                                    i+1,
                                                    current_time.strftime('%Y%m%d'),
                                                    current_time.hour*3600+current_time.second)))
        # Convert EnKF states back to VIC states
        class_states = States(ds)
        ds_updated = class_states.convert_new_EnKFstates_sm_to_VICstates(da_x_updated.loc[:, :, :, i])
        # Save VIC states to netCDF file
        ds_updated.to_netcdf(os.path.join(out_vic_state_dir,
                                          'state.ens{}.nc'.format(i+1)),
                             format='NETCDF4_CLASSIC')
    
    return da_x_updated


def perturb_forcings_ensemble(N, orig_forcing, year, dict_varnames, prec_std,
                              out_forcing_dir):
    ''' Perturb forcings for all ensemble members
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    orig_forcing: <class 'Forcings'>
        Original (unperturbed) VIC forcings
    year: <int>
        Year of forcing
    dict_varnames: <dict>
        A dictionary of forcing names in nc file;
        e.g., {'PREC': 'prcp'; 'AIR_TEMP': 'tas'}
    prec_std: <float>
        Standard deviation of the precipitation perturbing multiplier
    out_forcing_dir: <str>
        Directory for output perturbed forcings;
        File names will be: forc.ens<i>.nc, where <i> is ensemble index, 1, ..., N
    
    Require
    ----------
    os
    '''
    
    # Loop over each ensemble member
    for i in range(N):
        # Perturb PREC
        ds_perturbed = orig_forcing.perturb_prec_lognormal(
                                            varname=dict_varnames['PREC'],
                                            std=prec_std)
        # Save to nc file
        ds_perturbed.to_netcdf(os.path.join(out_forcing_dir,
                                            'forc.ens{}.{}.nc'.format(i+1, year)),
                               format='NETCDF4_CLASSIC')


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


def perturb_soil_moisture_states_ensemble(N, states_to_perturb_dir, global_path,
                                          sigma_percent, out_states_dir):
    ''' Perturb all soil_moisture states for each ensemble member
    
    Parameters
    ----------
    N: <int>
        Number of ensemble members
    states_to_perturb_dir: <str>
        Directory for VIC state files to perturb.
        File names: state.ens<i>.nc, where <i> is ensemble index 1, ..., N
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract soil parameter file info)
    sigma_percent: <float>
        Percentage of the maximum state value to perturb; sigma_percent will be used
        as the standard deviation of the Gaussian noise added (e.g., sigma_percent = 5
        for 5% of maximum soil moisture perturbation)
    out_states_dir: <str>
        Directory for output perturbed VIC state files;
        File names: state.ens<i>.nc, where <i> is ensemble index 1, ..., N
    
    Require
    ----------
    os
    class States
    '''
    
    for i in range(N):
        # Load in original state file
        class_states = States(xr.open_dataset(
                                os.path.join(states_to_perturb_dir,
                                             'state.ens{}.nc'.format(i+1))))
        # Perturb
        ds_perturbed = class_states.perturb_soil_moisture_Gaussian(
                                global_path, sigma_percent)
        # Save perturbed state file
        ds_perturbed.to_netcdf(os.path.join(out_states_dir,
                                            'state.ens{}.nc'.format(i+1)),
                               format='NETCDF4_CLASSIC')


def propagate(start_time, end_time, vic_exe, vic_global_template_file,
                       vic_model_steps_per_day, init_state_nc, out_state_basepath,
                       out_history_dir, out_history_fileprefix,
                       out_global_basepath, out_log_dir,
                       forcing_basepath):
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
        Basepath of output states; ".YYYYMMDD_SSSSS.nc" will be appended
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
    returncode = vic_exe.run(global_file, logdir=out_log_dir)
    check_returncode(returncode, expected=0)


def perturb_soil_moisture_states(states_to_perturb_nc, global_path,
                                 sigma_percent, out_states_nc):
    ''' Perturb all soil_moisture states

    Parameters
    ----------
    states_to_perturb_nc: <str>
        Path of VIC state netCDF file to perturb.
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract soil parameter file info)
    sigma_percent: <float>
        Percentage of the maximum state value to perturb; sigma_percent will be used
        as the standard deviation of the Gaussian noise added (e.g., sigma_percent = 5
        for 5% of maximum soil moisture perturbation)
    out_states_nc: <str>
        Path of output perturbed VIC state netCDF file

    Require
    ----------
    os
    class States
    '''
    
    # Load in original state file
    class_states = States(xr.open_dataset(states_to_perturb_nc))

    # Perturb
    ds_perturbed = class_states.perturb_soil_moisture_Gaussian(
                                        global_path, sigma_percent)
    # Save perturbed state file
    ds_perturbed.to_netcdf(out_states_nc,
                           format='NETCDF4_CLASSIC')


def concat_vic_history_files(list_history_nc):
    ''' Concatenate short-periods of VIC history files into one; if the time period of
        the next file overlaps that of the current file, the next-file values will be
        used
    
    list_history_nc: <list>
        A list of history files to be concatenated, in order
    '''
    
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
            trunc_time_point = times_nest[0] - pd.DateOffset(seconds=2) 
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
    soil_nc = find_global_param_value(global_param, 'SOIL')
    ds_soil = xr.open_dataset(soil_nc, decode_cf=False)
    
    # Calculate maximum soil moisture for each layer
    # Dimension: [nlayer, lat, lon]
    da_depth = ds_soil['depth']  # [m]
    da_bulk_density = ds_soil['bulk_density']  # [kg/m3]
    da_soil_density = ds_soil['soil_density']  # [kg/m3]
    da_porosity = 1 - da_bulk_density / da_soil_density
    da_max_moist = da_depth * da_porosity * 1000  # [mm]

    return da_max_moist


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
    
    # Number of files
    N = len(list_state_nc)
    
    # Calculate ensemble mean
    list_ds = []
    for state_nc in list_state_nc:
        list_ds.append(xr.open_dataset(state_nc))
    ds_mean = sum(list_ds) / N

    # Replace some variables with those from the first state file to
    # prevent precision issues
    ds_first_state = list_ds[0]
    ds_mean['layer'] = ds_first_state['layer']
    ds_mean['dz_node'] = ds_first_state['dz_node']
    ds_mean['node_depth'] = ds_first_state['node_depth']
    
    # Write to output netCDF file
    ds_mean.to_netcdf(out_state_nc, format='NETCDF4_CLASSIC')
    
    return out_state_nc


def run_vic_assigned_states(start_time, end_time, vic_exe, init_state_nc, dict_assigned_state_nc,
                            global_template, vic_forcing_basepath, vic_model_steps_per_day,
                            output_global_root_dir, output_state_root_dir, output_vic_history_root_dir,
                            output_vic_log_root_dir):
    ''' Run VIC with assigned initial states and other assigned state files during the simulation time
    
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
    output_state_root_dir: <str>
        Directory for VIC output state files
    output_vic_history_root_dir: <str>
        Directory for VIC output history files
    output_vic_log_root_dir: <str>
        Directory for VIC output log files
    
    Returns
    ----------
    list_history_files: <list>
        A list of all output history file paths in order
    
    Require
    ----------
    OrderedDict
    generate_VIC_global_file
    check_returncode
    '''
    
    list_history_files = []  # A list of resulted history file paths
    
    # --- Run VIC from start_time to the first assigned state time --- #
    run_start_time = start_time
    run_end_time = list(dict_assigned_state_nc.keys())[0]
    print('\tRunning VIC from ', run_start_time, 'to', run_end_time)
    propagate(start_time=run_start_time, end_time=run_end_time,
              vic_exe=vic_exe, vic_global_template_file=global_template,
              vic_model_steps_per_day=vic_model_steps_per_day,
              init_state_nc=init_state_nc,
              out_state_basepath=os.path.join(output_state_root_dir, 'state.tmp'),
                            # 'tmp' indicates this output state file is not usefull, since it will be replaced
                            # by assinged states
              out_history_dir=output_vic_history_root_dir,
              out_history_fileprefix='history',
              out_global_basepath=os.path.join(output_global_root_dir, 'global'),
              out_log_dir=output_vic_log_root_dir,
              forcing_basepath=vic_forcing_basepath)
    list_history_files.append(os.path.join(
                    output_vic_history_root_dir,
                    'history.{}-{:05d}.nc'.format(
                            run_start_time.strftime('%Y-%m-%d'),
                            run_start_time.hour*3600+run_start_time.second)))
    
    # --- Run VIC from each assigned state time to the next (or to end_time) --- #
    for t, time in enumerate(dict_assigned_state_nc.keys()):
        # --- Determine last, current and next measurement time points --- #
        last_time = time
        current_time = last_time + pd.DateOffset(hours=24/vic_model_steps_per_day)
        if t == len(dict_assigned_state_nc)-1:  # if this is the last measurement time
            next_time = end_time
        else:  # if not the last measurement time
            next_time = list(dict_assigned_state_nc.keys())[t+1]
        # If current_time > next_time, do not propagate (we already reach the end of the simulation)
        if current_time > next_time:
            break
        print('\tRunning VIC from ', current_time, 'to', next_time)
        
        # --- Propagate to the next time from assigned initial states --- #
        state_nc = dict_assigned_state_nc[last_time]
        propagate(start_time=current_time, end_time=next_time,
                  vic_exe=vic_exe, vic_global_template_file=global_template,
                  vic_model_steps_per_day=vic_model_steps_per_day,
                  init_state_nc=state_nc,
                  out_state_basepath=os.path.join(output_state_root_dir, 'state.tmp'),
                            # 'tmp' indicates this output state file is not usefull, since it will be replaced
                            # by assinged states
                  out_history_dir=output_vic_history_root_dir,
                  out_history_fileprefix='history',
                  out_global_basepath=os.path.join(output_global_root_dir, 'global'),
                  out_log_dir=output_vic_log_root_dir,
                  forcing_basepath=vic_forcing_basepath)
        list_history_files.append(os.path.join(
                    output_vic_history_root_dir,
                    'history.{}-{:05d}.nc'.format(
                            current_time.strftime('%Y-%m-%d'),
                            current_time.hour*3600+current_time.second)))
    
    return list_history_files


def rmse(df, var_true, var_est):
    ''' Calculates RMSE of an estimated variable compared to the truth variable
    
    Parameters
    ----------
    df: <pd.DataFrame>
        A dataframe containing the two time series
    var_true: <str>
        Name of the truth variable in the df
    var_est: <str>
        Name of the estimated variable in the df
    
    Returns
    ----------
    rmse: <float>
        Root mean square error
    
    Require
    ----------
    numpy
    '''
    
    rmse = np.sqrt(sum((df[var_est] - df[var_true])**2) / len(df))
    return rmse

