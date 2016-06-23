
import numpy as np
import pandas as pd
import os
import string
from collections import OrderedDict
import xarray as xr


class VicStates(object):
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
    
    
    def add_gaussian_white_noise_soil_moisture(self, P):
        ''' Add Gaussian noise for all active grid cells

        Parameters
        ----------
        P: <float> [nlayer*nlayer]
            Covariance matrix of the Gaussian noise to be added
            
        Returns
        ----------
        ds: <xr.dataset>
            A dataset of VIC states, with purterbed soil moisture
        '''

        # Extract number of soil layers
        nlayer = len(self.ds['nlayer'])
        
        # Generate random noise for the whole field
        noise = np.random.multivariate_normal(
                        np.zeros(nlayer), P,
                        size=len(self.ds['veg_class'])*\
                             len(self.ds['snow_band'])*\
                             len(self.ds['lat'])*\
                             len(self.ds['lon']))
        noise_reshape = noise.reshape([len(self.ds['veg_class']),
                                       len(self.ds['snow_band']),
                                       nlayer,
                                       len(self.ds['lat']),
                                       len(self.ds['lon'])])
        # Add noise to soil moisture field
        ds = self.ds.copy()
        ds['Soil_moisture'] = self.ds['Soil_moisture'] + noise_reshape
        return ds


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
    P0: <np.array>, 2-D, [n*n]
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
    n = np.shape(P0)[0]  # numer of states
    m = 1  # number of measurements
    n_time = len(da_meas[da_meas_time_var])  # number of measurement time points
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
    class_states = VicStates(ds_states)
    # Set up initial state subdirectories
    init_state_dir_name = 'init.{}_{:05d}'.format(
                                    start_time.strftime('%Y%m%d'),
                                    start_time.hour*3600+start_time.second)
    init_state_dir = setup_output_dirs(
                            output_vic_state_root_dir,
                            mkdirs=[init_state_dir_name])[init_state_dir_name]
    # For each ensemble member, add Gaussian noise to sm states with covariance P0,
    # and save each ensemble member states
    for i in range(N):
        ds = class_states.add_gaussian_white_noise_soil_moisture(P0)
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
        da_x, da_y = get_soil_moisture_and_estimated_meas_all_ensemble(
                            N,
                            state_dir=state_dir_before_update,
                            state_time=current_time)
        da_K = calculate_gain_K_whole_field(da_x, da_y)
        
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


def calculate_gain_K(n, m, x, y):
    ''' This function calculates Kalman gain K from ensemble.
    
    Parameters
    ----------
    n: <int>
        Number of states
    m: <int>
        Number of measurements
    x: <np.array> [n*N]
        An array of forecasted ensemble states (before updated)
    y: <np.array> [m*N]
        An array of forecasted ensemble measurement estimates (before updated);
        (y = Hx)
    
    Returns
    ----------
    K: <np.array> [n*m]
        Gain K
    
    Require
    ----------
    numpy
    '''
    
    # Pxy = cov(x, y.transpose); size = [n*m]; divided by (N-1)
    Pxy = np.cov(x, y)[:n, n:]
    # Pyy = cov(y, y.transpose); size = [m*m]; divided by (N-1)
    Pyy = np.cov(y)
    # K = Pxy * (Pyy)-1
    if m == 1:  # if m = 1
        K = Pxy / Pyy
    else:  # if m > 1
        K = np.dot(Pxx, np.linalg.inv(Pyy))

    return K


def get_soil_moisture_and_estimated_meas_all_ensemble(N, state_dir, state_time):
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

    Returns
    ----------
    da_x: <xr.DataArray>
        Soil moisture states of all ensemble members;
        Dimension: [veg_class, snow_band, lat, lon, n, N]
    da_y: <xr.DataArray>
        Estimated measurement of all ensemble members (= top-layer soil moisture);
        Dimension: [eg_class, snow_band, lat, lon, m, N]
    
    Require
    ----------
    xarray
    os
    '''
    
    # --- Extract dimensions --- #
    # n = nlayer, number of states
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
    
    # --- Initialize da for states and measurement estimates --- #
    # Initialize states x [veg_class, snow_band, lat, lon, n, N]
    x = np.empty([len(veg_class), len(snow_band), len(lat), len(lon), len(nlayer), N])
    x[:] = np.nan
    da_x = xr.DataArray(x,
                        coords=[veg_class, snow_band, lat, lon, nlayer, range(1, N+1)],
                        dims=['veg_class', 'snow_band', 'lat', 'lon', 'n', 'N'])
    # Initialize measurement estimates y [eg_class, snow_band, lat, lon, m, N]
    y = np.empty([len(veg_class), len(snow_band), len(lat), len(lon), 1, N])
    y[:] = np.nan
    da_y = xr.DataArray(y,
                        coords=[veg_class, snow_band, lat, lon, [1], range(1, N+1)],
                        dims=['veg_class', 'snow_band', 'lat', 'lon', 'm', 'N'])
    
    # --- Loop over each ensemble member --- #
    for i in range(N):
        # --- Load state file --- #
        state_name = 'state.ens{}.{}_{}.nc'.format(i+1,
                                                   state_time.strftime('%Y%m%d'),
                                                   state_time.hour*3600+state_time.second)
        ds = xr.open_dataset(os.path.join(state_dir, state_name))
        
        # --- Fill x and y data in --- #
        # Fill in states x
        for j in nlayer:
            # Fill in states x
            da_x.loc[:, :, :, :, j, i+1] = ds['Soil_moisture'].loc[:, :, j, :, :]
        # Fill in measurement estimates y (here it is simply top-layer sm)
        da_y.loc[:, :, :, :, 1, i+1] = ds['Soil_moisture'].loc[:, :, nlayer[0].values, :, :]
        
    return da_x, da_y


def calculate_gain_K_whole_field(da_x, da_y):
    ''' This function calculates gain K over the whole field.
    
    Parameters
    ----------
    da_x: <xr.DataArray>
        Soil moisture states of all ensemble members;
        As returned from get_soil_moisture_and_estimated_meas_all_ensemble;
        Dimension: [veg_class, snow_band, lat, lon, n, N]
    da_y: <xr.DataArray>
        Estimated measurement of all ensemble members (= top-layer soil moisture);
        As returned from get_soil_moisture_and_estimated_meas_all_ensemble;
        Dimension: [eg_class, snow_band, lat, lon, m, N]
        
    Returns
    ----------
    da_K: <xr.DataArray>
        Gain K for the whole field
        Dimension: [veg_class, snow_band, lat, lon, n, m], 
        where [n, m] is the Kalman gain K
    
    require
    ----------
    xarray
    calculate_gain_K
    numpy
    '''

    # --- Extract dimensions --- #
    veg_class = da_x['veg_class']
    snow_band = da_x['snow_band']
    lat_coord = da_x['lat']
    lon_coord = da_x['lon']
    n_coord = da_x['n']
    m_coord = da_y['m']
    
    # --- Initialize da_K --- #
    K = np.empty([len(veg_class), len(snow_band), len(lat_coord), len(lon_coord),
                  len(n_coord), len(m_coord)])
    K[:] = np.nan
    da_K = xr.DataArray(K,
                        coords=[veg_class, snow_band, lat_coord, lon_coord,
                                n_coord, m_coord],
                        dims=['veg_class', 'snow_band', 'lat', 'lon', 'n', 'm'])
    
    # --- Calculate gain K for the whole field --- #
    for veg in veg_class:
            for snow in snow_band:
                for lat in lat_coord:
                    for lon in lon_coord:
                        # Skip inactive tiles
                        if np.isnan(da_x.loc[veg, snow, lat, lon,
                                             n_coord[0].values,
                                             m_coord[0].values]) == True:
                            continue
                        # Calculate gain K
                        K = calculate_gain_K(len(n_coord), len(m_coord),
                                             da_x.loc[veg, snow, lat, lon, :, :],
                                             da_y.loc[veg, snow, lat, lon, :, :])
                        # Fill K in da_K
                        da_K.loc[veg, snow, lat, lon, :, :] = K
    
    return da_K
