
import numpy as np
import pandas as pd
import os
import string
from collections import OrderedDict
import xarray as xr


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
        EnKF_states = self.ds['Soil_moisture'].values.reshape(
                                                [n, len(lat), len(lon)])
        
        # roll the 'n' dimension to after lat and lon, and fill in da
        EnKF_states = np.rollaxis(EnKF_states, 0, 3) 
        da_EnKF[:] = EnKF_states
        
        # Save as self.da_EnKF
        self.da_EnKF = da_EnKF
        
        return self.da_EnKF
    
    
    def convert_EnKFstates_sm_to_VICstates(self, da_EnKF):
        ''' This function converts a EnKF states da (soil moisture states) back to the
            original VIC states ds (self.ds) - as a returned ds, withouth changing self.ds
        
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
        ds['Soil_moisture'][:] = EnKF_states_reshape
        
        return ds
        
    
    def add_gaussian_white_noise_states(self, P):
        ''' Add Gaussian noise for the whole field.
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
        
        # Put the perturbed soil moisture states to back to VIC states ds
        ds = self.convert_EnKFstates_sm_to_VICstates(da_perturbed)

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


def EnKF_VIC(N, start_time, end_time, init_state_basepath, P0, da_meas,
             da_meas_time_var, vic_exe, vic_global_template,
             vic_model_steps_per_day, output_vic_global_root_dir,
             output_vic_state_root_dir, output_vic_history_root_dir,
             output_vic_log_root_dir):
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
        Initial state directory and file name prefix (excluding '.YYMMDD_SSSSS.nc')
    P0: <float>
        Initial state error matrix
    da_meas: <xr.DataArray> [time*lat*lon]
        DataArray of measurements (currently, must be only 1 variable of measurement);
        Measurements should already be truncated (if needed) so that they are all within the
        EnKF run period
    da_meas_time_var: <str>
        Time variable name in da_meas
    vic_exe: <class 'VIC'>
        VIC run class
    vic_global_template: <str>
        VIC global file template
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
    
    # --- Initialize ---#
    # Load initial state file
    ds_states = xr.open_dataset('{}.{}_{:05d}.nc'.format(
                                        init_state_basepath,
                                        start_time.strftime('%Y%m%d'),
                                        start_time.hour*3600+start_time.second))
    class_states = States(ds_states)
    
    # Determine the number of EnKF states, n
    n = len(class_states.da_EnKF['n'])
    # Set up initial state subdirectories
    init_state_dir_name = 'init.{}_{:05d}'.format(
                                    start_time.strftime('%Y%m%d'),
                                    start_time.hour*3600+start_time.second)
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
    
    # --- Propagate (run VIC) until the first measurement time point ---#    
    # Determine VIC run period
    vic_run_start_time = start_time
    vic_run_end_time = pd.to_datetime(da_meas[da_meas_time_var].values[0])
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
                       vic_exe = vic_exe,
                       vic_global_template_file=vic_global_template,
                       vic_model_steps_per_day=vic_model_steps_per_day,
                       init_state_dir=init_state_dir,
                       out_state_dir=out_state_dir,
                       out_history_dir=out_history_dir,
                       out_global_dir=out_global_dir,
                       out_log_dir=out_log_dir)
    
    # --- Run EnKF --- #
    # Initialize
    state_dir_before_update = out_state_dir
    
    # Loop over each measurement time point
    for k in range(n_time):
        
        # Determine current time
        current_time = pd.to_datetime(da_meas[da_meas_time_var].values[k])
        
        # Calculate gain K
        da_x, da_y_est = get_soil_moisture_and_estimated_meas_all_ensemble(
                                N,
                                state_dir=state_dir_before_update,
                                state_time=current_time, 
                                da_tile_frac=da_tile_frac)
        da_K = calculate_gain_K_whole_field(da_x, da_y_est)

        return


def generate_VIC_global_file(global_template_path, model_steps_per_day,
                             start_time, end_time, init_state, vic_state_basepath,
                             vic_history_file_basepath, output_global_basepath):
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
    vic_history_file_basepath: <str>
        Output file basepath
        ".<start_time>_<end_date>.nc" will be appended,
            where <start_time> is in '%Y%m%d-%H%S',
                  <end_date> is in '%Y%m%d' (since VIC always runs until the end of a date)
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
                                     result_dir='{}.{}_{}.nc'.format(
                                            vic_history_file_basepath,
                                            start_time.strftime('%Y%m%d-%H%S'),
                                            end_time.strftime('%Y%m%d')))
    
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
    '''check return code given by VIC, raise error if appropriate'''
    if returncode == expected:
        return None
    elif returncode == default_vic_valgrind_error_code:
        raise VICValgrindError('Valgrind raised an error')
    else:
        raise VICReturnCodeError('VIC return code ({0}) does not match '
                                 'expected ({1})'.format(returncode, expected))


def propagate_ensemble(N, start_time, end_time, vic_exe, vic_global_template_file,
                       vic_model_steps_per_day, init_state_dir, out_state_dir,
                       out_history_dir, out_global_dir, out_log_dir):
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
        
    Require
    ----------
    generate_VIC_global_file
    '''
    
    # --- Loop over each ensemble member --- #
    for i in range(N):
        # Generate VIC global param file
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
                            vic_history_file_basepath=os.path.join(
                                        out_history_dir,
                                        'history.ens{}'.format(i+1)),
                            output_global_basepath=os.path.join(
                                        out_global_dir,
                                        'global.ens{}'.format(i+1)))
        # Run VIC
#        returncode = vic_exe.run(global_file, logdir=out_log_dir)
#        check_returncode(returncode, expected=0)


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
                                        class_states.ds['Soil_moisture'], da_tile_frac)
        
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


def calculate_gain_K(x, y_est):
    ''' This function calculates Kalman gain K from ensemble.
    
    Parameters
    ----------
    x: <np.array> [n*N]
        An array of forecasted ensemble states (before updated)
    y_est: <np.array> [m*N]
        An array of forecasted ensemble measurement estimates (before updated);
        (y_est = Hx)
    
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
        K = Pxy / Pyy
    else:  # if m > 1
        K = np.dot(Pxx, np.linalg.inv(Pyy))

    return K


def calculate_gain_K_whole_field(da_x, da_y_est):
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
                                 da_y_est.loc[lat, lon, :, :])
            # Fill K in da_K
            da_K.loc[lat, lon, :, :] = K
    
    return da_K



