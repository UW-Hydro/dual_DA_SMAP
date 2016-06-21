
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


def EnKF_VIC(N, start_time, end_time, init_state_basepath, P0, da_meas, da_meas_time_var, vic_exe,
             vic_global_template, vic_model_steps_per_day, output_vic_global_dir,
             output_vic_state_dir, output_vic_result_dir, output_vic_log_dir):
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
    output_vic_global_dir: <str>
        Directory for VIC global files
    output_vic_state_dir: <str>
        Directory for VIC output state files
    output_vic_result_dir: <str>
        Directory for VIC output result files
    output_vic_log_dir: <str>
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
    # Set up state subdirectories
    states_to_save_subdir = 'propagate.{}_{:05d}.nc'.format(
                                    start_time.strftime('%Y%m%d'),
                                    start_time.hour*3600+start_time.second)
    states_to_save_dir = setup_output_dirs(output_vic_state_dir,
                                           mkdirs=[states_to_save_subdir])
    # For each ensemble member, add Gaussian noise to sm states with covariance P0,
    # and save each ensemble member states
    for i in range(N):
        ds = class_states.add_gaussian_white_noise_soil_moisture(P0)
        ds.to_netcdf(os.path.join(states_to_save_dir[states_to_save_subdir],
                                  'state.ens{}'.format(i+1)),
                     format='NETCDF4_CLASSIC')
    raise
    
    
    # Generate ensemble of initial states
    x0_ens = np.random.multivariate_normal(x0, P0, size=N)  # [N*n]

    # --- Propagate (run VIC) until the first measurement time point ---#
    # Determine VIC run period
    vic_run_start_time = start_time
    vic_run_end_time = pd.to_datetime(da_meas[da_meas_time_var].values[0])
    # Generate VIC global param file (no initial state)
    global_file = generate_VIC_global_file(
                            global_template_path=vic_global_template,
                            model_steps_per_day=vic_model_steps_per_day,
                            start_time=vic_run_start_time,
                            end_time=vic_run_end_time,
                            init_state="# INIT_STATE",
                            vic_state_basepath=os.path.join(output_vic_state_dir,
                                                            'state'),
                            vic_history_file_basepath=os.path.join(
                                        output_vic_result_dir, 'history'),
                            output_global_basepath=os.path.join(
                                        output_vic_global_dir, 'global'))

    # Propagate (run VIC)
#    returncode = vic_exe.run(global_file, logdir=output_vic_log_dir)
#    check_returncode(returncode, expected=0)
    
    # --- Run EnKF --- #
    # loop over each measurement time point
    #for k in range(n_time):
        
    

    #print(np.random.normal(0, 1, size=5))


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

