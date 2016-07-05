
import sys
import numpy as np
import xarray as xr
import os
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

from tonic.models.vic.vic import VIC
from tonic.io import read_config, read_configobj
from da_utils import (EnKF_VIC, setup_output_dirs, generate_VIC_global_file,
                      check_returncode, propagate, calculate_ensemble_mean_states,
                      run_vic_assigned_states, concat_vic_history_files)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Set random generation seed
# ============================================================ #
np.random.seed(cfg['CONTROL']['seed'])


# ============================================================ #
# Prepare output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_basdir']),
                         mkdirs=['global', 'history', 'states', 'forcings',
                                 'logs', 'plots'])


# ============================================================ #
# Prepare VIC exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))


# ============================================================ #
# VIC model spinup
# VIC will be run for a spinup period to get x0
# ============================================================ #
# --- Determine VIC spinup run period --- #
vic_run_start_time = pd.to_datetime(cfg['VIC']['spinup_start_time'])
vic_run_end_time = pd.to_datetime(cfg['EnKF']['start_time']) -\
                   pd.DateOffset(hours=24/cfg['VIC']['model_steps_per_day'])

print('Running spinup time: ', vic_run_start_time, 'to', vic_run_end_time,
       '...')

# --- Generate VIC global param file (no initial state) --- #
# Specify forcing file and output history file name
replace = OrderedDict([('FORCING1', os.path.join(
                                        cfg['CONTROL']['root_dir'],
                                        cfg['FORCINGS']['orig_forcing_nc_basepath'])),
                       ('OUTFILE', 'history.spinup')])
global_file = generate_VIC_global_file(
                        global_template_path=os.path.join(cfg['CONTROL']['root_dir'],
                                                          cfg['VIC']['vic_global_template']),
                        model_steps_per_day=cfg['VIC']['model_steps_per_day'],
                        start_time=vic_run_start_time,
                        end_time=vic_run_end_time,
                        init_state="# INIT_STATE",
                        vic_state_basepath=os.path.join(dirs['states'], 'state.spinup'),
                        vic_history_file_dir=dirs['history'],
                        replace=replace,
                        output_global_basepath=os.path.join(dirs['global'],
                                                            'global.spinup'))

# --- Prepare log directory --- #
log_dir = setup_output_dirs(dirs['logs'],
                            mkdirs=['spinup'])['spinup']

# --- Run VIC --- #
returncode = vic_exe.run(global_file, logdir=log_dir)
check_returncode(returncode, expected=0)


# ============================================================ #
# Open-loop run
# ============================================================ #
# --- Determine open-loop run period --- #
vic_run_start_time = pd.to_datetime(cfg['EnKF']['start_time'])
vic_run_end_time = pd.to_datetime(cfg['EnKF']['end_time'])

print('Running open-loop: ', vic_run_start_time, 'to', vic_run_end_time,
       '...')

# --- Run VIC (unperturbed forcings and states) --- #
# Identify initial state time (must be one time step before start_time)
init_state_time = vic_run_start_time -\
                  pd.DateOffset(hours=24/cfg['VIC']['model_steps_per_day'])
# Prepare log sub-directory
out_log_dir = setup_output_dirs(dirs['logs'], mkdirs=['openloop'])['openloop']
propagate(start_time=vic_run_start_time, end_time=vic_run_end_time,
          vic_exe=vic_exe,
          vic_global_template_file=os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['VIC']['vic_global_template']),
           vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
           init_state_nc=os.path.join(
                                dirs['states'],
                                'state.spinup.{}_{:05d}.nc'.format(
                                        init_state_time.strftime('%Y%m%d'),
                                        init_state_time.hour*3600+init_state_time.second)),
           out_state_basepath=os.path.join(dirs['states'], 'state.openloop'),
           out_history_dir=dirs['history'],
           out_history_fileprefix='history.openloop',
           out_global_basepath=os.path.join(dirs['global'], 'global.openloop'),
           out_log_dir=out_log_dir,
           forcing_basepath=os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['FORCINGS']['orig_forcing_nc_basepath']))


# ============================================================ #
# Prepare and run EnKF
# ============================================================ #
print('Preparing for running EnKF...')

# --- Load and process measurement data --- #
print('\tLoading measurement data...')
# Load measurement data
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
# Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(cfg['EnKF']['start_time'], cfg['EnKF']['end_time']))
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])

# --- Process VIC forcing names and perturbation parameters --- #
# Construct forcing variable name dictionary
dict_varnames = {}
dict_varnames['PREC'] = cfg['FORCINGS']['PREC']

# --- Prepare measurement error covariance matrix R [m*m] --- #
R = np.array([[cfg['EnKF']['R']]])

# --- Run EnKF --- #
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])
print('Start running EnKF for ', start_time, 'to', end_time, '...')

EnKF_VIC(N=cfg['EnKF']['N'],
         start_time=start_time,
         end_time=end_time,
         init_state_basepath=os.path.join(dirs['states'], 'state.spinup'),
         P0=cfg['EnKF']['P0'],
         R=R,
         da_meas=da_meas,
         da_meas_time_var='time',
         vic_exe=vic_exe,
         vic_global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['VIC']['vic_global_template']),
         vic_forcing_orig_basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['FORCINGS']['orig_forcing_nc_basepath']),
         vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
         output_vic_global_root_dir=dirs['global'],
         output_vic_state_root_dir=dirs['states'],
         output_vic_history_root_dir=dirs['history'],
         output_vic_forcing_root_dir=dirs['forcings'],
         output_vic_log_root_dir=dirs['logs'],
         dict_varnames=dict_varnames, prec_std=cfg['FORCINGS']['prec_std'],
         state_perturb_sigma_percent=cfg['EnKF']['state_perturb_sigma_percent'])


# ============================================================ #
# Calculate the ensemble-mean of the updated states
# ============================================================ #
print('Calculating ensemble-mean of the updates states...')

N=cfg['EnKF']['N']  # number of ensemble members

# --- Calculate ensemble-mean for the initial time point --- #
init_time = pd.to_datetime(cfg['EnKF']['start_time'])
# Create a list of state file nc paths
state_dir = os.path.join(dirs['states'], 'init.{}_{:05d}'.format(
                                init_time.strftime('%Y%m%d'),
                                init_time.hour*3600+init_time.second))
list_state_nc = []
for i in range(N):
    list_state_nc.append(os.path.join(state_dir, 'state.ens{}.nc'.format(i+1)))
# Calculate ensemble-mean states
init_state_mean_nc = calculate_ensemble_mean_states(list_state_nc,
                                               out_state_nc=os.path.join(state_dir, 'state.ens_mean.nc'))

# Loop over each measurement time point of updates states
dict_assigned_state_nc = OrderedDict()  #  An ordered dict of state times and nc files after the initial time
for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
    state_time = pd.to_datetime(time)
    # Create a list of state file nc paths
    state_dir = os.path.join(dirs['states'], 'updated.{}_{:05d}'.format(
                                state_time.strftime('%Y%m%d'),
                                state_time.hour*3600+state_time.second))
    list_state_nc = []
    for i in range(N):
        list_state_nc.append(os.path.join(state_dir, 'state.ens{}.nc'.format(i+1)))
    # Calculate ensemble-mean states
    dict_assigned_state_nc[time] = calculate_ensemble_mean_states(
                                        list_state_nc,
                                        out_state_nc=os.path.join(state_dir, 'state.ens_mean.nc'))


# ============================================================ #
# Post-process fluxes - run VIC with ensemble-mean updated
# states from EnKF start_time to EnKF end_time
# ============================================================ #
print('Post-process - run VIC with ensemble-mean updated states...')
# Set up output sub-directories
out_global_dir = setup_output_dirs(dirs['global'], mkdirs=['postprocess_ens_mean_updated_states'])\
                                                    ['postprocess_ens_mean_updated_states']
out_state_dir = setup_output_dirs(dirs['states'], mkdirs=['postprocess_ens_mean_updated_states_tmp'])\
                                                    ['postprocess_ens_mean_updated_states_tmp']
out_history_dir = setup_output_dirs(dirs['history'], mkdirs=['postprocess_ens_mean_updated_states'])\
                                                    ['postprocess_ens_mean_updated_states']
out_log_dir = setup_output_dirs(dirs['logs'], mkdirs=['postprocess_ens_mean_updated_states'])\
                                                    ['postprocess_ens_mean_updated_states']

# Run VIC with assinged states
list_history_files = run_vic_assigned_states(
                        start_time=start_time, end_time=end_time,
                        vic_exe=vic_exe, init_state_nc=init_state_mean_nc,
                        dict_assigned_state_nc=dict_assigned_state_nc,
                        global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                                     cfg['VIC']['vic_global_template']),
                        vic_forcing_basepath=os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['FORCINGS']['orig_forcing_nc_basepath']),  # Original (unperturbed) forcing
                        vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
                        output_global_root_dir=out_global_dir,
                        output_state_root_dir=out_state_dir,
                        output_vic_history_root_dir=out_history_dir,
                        output_vic_log_root_dir=out_log_dir)

# Concatenate all history files
ds_concat = concat_vic_history_files(list_history_files)
out_history_postprocess_dir = out_history_dir
hist_ens_mean_post = os.path.join(
                        out_history_postprocess_dir,
                        'history.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                start_time.strftime('%Y%m%d'),
                                start_time.hour*3600+start_time.second,
                                end_time.strftime('%Y%m%d'),
                                end_time.hour*3600+end_time.second))
ds_concat.to_netcdf(hist_ens_mean_post, format='NETCDF4_CLASSIC')

