
''' This script post processes EnKF updated soil moisture states and SMART
    corrected rainfall. Specifically:
        1) Take average of all ensemble updated states
        2) Insert the updated states, and use corrected rainfall forcing to run VIC

    Usage:
        $ python postprocess_EnKF.py <config_file> <nproc> <mpi_proc>
'''

from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import xarray as xr
import sys
import multiprocessing as mp

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic import VIC
from da_utils import (setup_output_dirs, run_vic_assigned_states,
                      concat_vic_history_files,
                      calculate_ensemble_mean_states,
                      Forcings)

# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Read number of processors for python multiprocessing
nproc = int(sys.argv[2])

# Read number of processors for VIC MPI runs
mpi_proc = int(sys.argv[3])

# ============================================================ #
# Prepare VIC exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))


# ============================================================ #
# Process cfg data
# ============================================================ #
N = cfg['EnKF']['N']  # number of ensemble members
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])

start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Setup postprocess output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(
                            cfg['CONTROL']['root_dir'],
                            cfg['POSTPROCESS']['output_postprocess_basedir']),
                         mkdirs=['global', 'history', 'forcings',
                                 'logs', 'plots'])


# ============================================================ #
# Load data
# ============================================================ #
# --- Load measurement data --- #
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
# Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(start_time, end_time))
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])


# ============================================================ #
# Post-process fluxes - run VIC from EnKF_start_time to
# EnKF_end_time.
# States:
#   EnKF updated states; either keep the ensemble of updated
#   states, or only use the ensemble mean updated states
#   (specified in the cfg file).
# Forcings:
#   Specified prec forcing (other forcing variables are from orig. forcings)
# ============================================================ #
# ----------------------------------------------------------------- #
# --- Generate forcings for post-processing - replace prec data --- #
# ----------------------------------------------------------------- #
print('Replacing precip data...')
if cfg['POSTPROCESS']['if_ens_states'] == False and \
   cfg['POSTPROCESS']['if_ens_prec'] == True:
    print('Error: if if_ens_states = False, ' \
          'output_postprocess_basedir must be False as well.')
    exit()
if cfg['POSTPROCESS']['if_ens_prec'] == True:
    # !!!!!!!!! Haven't implemented ensemble precip forcing for postprocessing yet !!!!!
    pass
# Else if use one precip data for all ensemble members
else:
    for year in range(start_year, end_year+1):
        # Load prec data
        da_prec = xr.open_dataset('{}{}.nc'.format(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['POSTPROCESS']['prec_nc_basepath']),
                    year))[cfg['POSTPROCESS']['prec_varname']]
        # Load in orig forcings
        class_forcings_orig = Forcings(xr.open_dataset('{}{}.nc'.format(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['FORCINGS']['orig_forcing_nc_basepath']),
                    year)))
        # Replace prec
        ds_prec_replaced = class_forcings_orig.replace_prec(
                                cfg['FORCINGS']['PREC'],
                                da_prec)
        # Save replaced forcings to netCDF file
        ds_prec_replaced.to_netcdf(os.path.join(
                            dirs['forcings'],
                            'forc.post_prec.{}.nc'.format(year)),
                         format='NETCDF4_CLASSIC')


# ----------------------------------------------------------------- #
# --- Run VIC with assinged states --- #
# ----------------------------------------------------------------- #
print('Post-process - run VIC with updated states...')
# --- If take mean updated states --- #
if cfg['POSTPROCESS']['if_ens_states'] == False:
    # !!! This part hasn't been implemented correctly !!!
    pass
    list_history_files = run_vic_assigned_states(
        start_time=start_time, end_time=end_time,
        vic_exe=vic_exe, init_state_nc=init_state_mean_nc,
        dict_assigned_state_nc=dict_assigned_state_nc,
        global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                     cfg['VIC']['vic_global_template']),
        vic_forcing_basepath=os.path.join(
                dirs['forcings'], 'forc.post_prec.'),
        vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
        output_global_root_dir=dirs['global'],
        output_vic_history_root_dir=dirs['history'],
        output_vic_log_root_dir=dirs['logs'],
        mpi_proc=mpi_proc,
        mpi_exe=cfg['VIC']['mpi_exe'])
# --- Else if keep the ensemble of updated states --- #
else:
    if cfg['POSTPROCESS']['if_ens_prec'] == True:
        # !!!!!!!!! Haven't implemented ensemble precip forcing yet !!!!!
        pass

    # Else if use one precip data for all ensemble members
    else:
        # --- If single processor, do a regular run --- #
        if nproc == 1:
            for i in range(N):
                print('N = {}'.format(i+1))
                # --- Prepare some variables --- #
                # initial state nc
                init_state_nc = os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['OUTPUT']['output_EnKF_basedir'],
                    'states',
                    'init.{}_{:05d}'.format(
                            start_time.strftime('%Y%m%d'),
                            start_time.hour*3600+start_time.second),
                    'state.ens{}.nc'.format(i+1))
                # state file dict
                dict_assigned_state_nc = OrderedDict()
                for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
                    state_time = pd.to_datetime(time)
                    dict_assigned_state_nc[state_time] = os.path.join(
                        cfg['CONTROL']['root_dir'],
                        cfg['OUTPUT']['output_EnKF_basedir'],
                        'states',
                        'updated.{}_{:05d}'.format(
                                time.strftime('%Y%m%d'),
                                time.hour*3600+time.second),
                        'state.ens{}.nc'.format(i+1))
                # make subdirs for global, history and log files for
                # each ensemble member
                hist_subdir = setup_output_dirs(
                                    dirs['history'],
                                    mkdirs=['ens{}'.format(i+1)])\
                              ['ens{}'.format(i+1)]
                global_subdir = setup_output_dirs(
                                    dirs['global'],
                                    mkdirs=['ens{}'.format(i+1)])\
                              ['ens{}'.format(i+1)]
                log_subdir = setup_output_dirs(
                                    dirs['logs'],
                                    mkdirs=['ens{}'.format(i+1)])\
                              ['ens{}'.format(i+1)]
                # other variables
                global_template = os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['VIC']['vic_global_template'])
                vic_forcing_basepath = os.path.join(
                                dirs['forcings'], 'forc.post_prec.')
                # --- run VIC with assigned states --- #
                list_history_files = run_vic_assigned_states(
                    start_time=start_time,
                    end_time=end_time,
                    vic_exe=vic_exe,
                    init_state_nc=init_state_nc,
                    dict_assigned_state_nc=dict_assigned_state_nc,
                    global_template=global_template,
                    vic_forcing_basepath=vic_forcing_basepath,
                    vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
                    output_global_root_dir=global_subdir,
                    output_vic_history_root_dir=hist_subdir,
                    output_vic_log_root_dir=log_subdir,
                    mpi_proc=mpi_proc,
                    mpi_exe=cfg['VIC']['mpi_exe'])
        # --- If multiple processors --- #
        else:
            # Set up multiprocessing
            pool = mp.Pool(processes=nproc)

            for i in range(N):
                print('N = {}'.format(i+1))
                # --- Prepare some variables --- #
                # initial state nc
                init_state_nc = os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['OUTPUT']['output_EnKF_basedir'],
                    'states',
                    'init.{}_{:05d}'.format(
                            start_time.strftime('%Y%m%d'),
                            start_time.hour*3600+start_time.second),
                    'state.ens{}.nc'.format(i+1))
                # state file dict
                dict_assigned_state_nc = OrderedDict()
                for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
                    state_time = pd.to_datetime(time)
                    dict_assigned_state_nc[state_time] = os.path.join(
                        cfg['CONTROL']['root_dir'],
                        cfg['OUTPUT']['output_EnKF_basedir'],
                        'states',
                        'updated.{}_{:05d}'.format(
                                time.strftime('%Y%m%d'),
                                time.hour*3600+time.second),
                        'state.ens{}.nc'.format(i+1))
                # make subdirs for history outputs for each ensemble member
                hist_subdir = setup_output_dirs(
                                    dirs['history'],
                                    mkdirs=['ens{}'.format(i+1)])\
                              ['ens{}'.format(i+1)]
                global_subdir = setup_output_dirs(
                                    dirs['global'],
                                    mkdirs=['ens{}'.format(i+1)])\
                              ['ens{}'.format(i+1)]
                log_subdir = setup_output_dirs(
                                    dirs['logs'],
                                    mkdirs=['ens{}'.format(i+1)])\
                              ['ens{}'.format(i+1)]
                # other variables
                global_template = os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['VIC']['vic_global_template'])
                vic_forcing_basepath = os.path.join(
                                dirs['forcings'], 'forc.post_prec.')
                # --- run VIC with assigned states --- #
                pool.apply_async(run_vic_assigned_states,
                                 (start_time, end_time, vic_exe,
                                  init_state_nc, dict_assigned_state_nc,
                                  global_template, vic_forcing_basepath,
                                  cfg['VIC']['model_steps_per_day'],
                                  global_subdir,
                                  hist_subdir, log_subdir, mpi_proc,
                                  cfg['VIC']['mpi_exe']))
            # Finish multiprocessing
            if nproc > 1:
                pool.close()
                pool.join()



