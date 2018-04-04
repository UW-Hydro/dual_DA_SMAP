
''' This script performs EnKF assimilation of soil moisture into VIC states,
    based on original precipitation forcing.

    Usage:
        $ python run_data_assim.py config_file nproc mpi_proc debug
'''

import sys
import numpy as np
import xarray as xr
import os
import pandas as pd
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import numbers

from tonic.models.vic.vic import VIC
from tonic.io import read_config, read_configobj
from da_utils import (EnKF_VIC, setup_output_dirs, generate_VIC_global_file,
                      check_returncode, propagate, calculate_ensemble_mean_states,
                      run_vic_assigned_states, concat_vic_history_files,
                      calculate_sm_noise_to_add_magnitude,
                      calculate_max_soil_moist_domain,
                      convert_max_moist_n_state,
                      calculate_scale_n_whole_field,
                      calculate_cholesky_L,
                      extract_mismatched_grid_weight_info)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Number of processors for parallelizing ensemble runs
nproc = int(sys.argv[2])

# Number of processors for each VIC run
mpi_proc = int(sys.argv[3])

# Whether to print out debug temp files or not
debug = (sys.argv[4].lower() == 'true')

# Restart time (format: YYYYMMDD_SSSSS); DA should have already been
# propogated to this time, and will restart from the update step
# "None" for starting from scratch
restart = None if sys.argv[5].lower()=='none' else str(sys.argv[5])


# ============================================================ #
# Set random generation seed
# ============================================================ #
if restart is None:
    np.random.seed(cfg['CONTROL']['seed'])
else:  # If restart, load in the saved random state
    restart_time = pd.to_datetime(restart)
    random_state_file = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_EnKF_basedir'],
        'restart_log',
        '{}.after_update.random_state.pickle'.format(
            restart_time.strftime("%Y%m%d-%H-%M-%S")))
    with open(random_state_file, 'rb') as f:
        random_state = pickle.load(f)
    np.random.set_state(random_state)


# ============================================================ #
# Prepare output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_EnKF_basedir']),
                         mkdirs=['global', 'history', 'states',
                                 'logs', 'plots', 'temp', 'restart_log'])


# ============================================================ #
# Prepare VIC exe and MPI exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))
mpi_exe = cfg['VIC']['mpi_exe']


# ============================================================ #
# Prepare and run EnKF
# ============================================================ #
print('Preparing for running EnKF...')

# --- Process linear model substitute, if specified --- #
if 'LINEAR_MODEL' in cfg:
    linear_model = True
    prec_varname = cfg['LINEAR_MODEL']['prec_varname']
    dict_linear_model_param={'r1': cfg['LINEAR_MODEL']['r1'],
                             'r2': cfg['LINEAR_MODEL']['r2'],
                             'r3': cfg['LINEAR_MODEL']['r3'],
                             'r12': cfg['LINEAR_MODEL']['r12'],
                             'r23': cfg['LINEAR_MODEL']['r23']}
else:
    linear_model = False

# --- Load and process measurement data --- #
print('\tLoading measurement data...')
# Load measurement data
if not linear_model:
    ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                                cfg['EnKF']['meas_nc']))
else:
    ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                                cfg['LINEAR_MODEL']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
# Only select out the period within the EnKF run period
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
da_meas = da_meas_orig.sel(time=slice(cfg['EnKF']['start_time'], cfg['EnKF']['end_time']))
# If the first measurement time point is the same as start_time, exclude this point
if pd.to_datetime(da_meas['time'][0].values) == start_time:
    da_meas = da_meas[1:, :, :]
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])

# --- Prepare measurement error covariance matrix R [lat, lon, m, m] --- #
# If input is a numerical number, assign spatial constant R values
if isinstance(cfg['EnKF']['R'], numbers.Number):
    R = np.empty([len(da_meas['lat']), len(da_meas['lon']), 1, 1])
    R[:] = cfg['EnKF']['R']
# If input is an xr.Dataset
else:
    if cfg['EnKF']['R_vartype'] == 'R':
        R = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'], cfg['EnKF']['R']))\
            [cfg['EnKF']['R_varname']].values
    elif cfg['EnKF']['R_vartype'] == 'std':
        R = np.square(xr.open_dataset(
            os.path.join(cfg['CONTROL']['root_dir'], cfg['EnKF']['R']))\
            [cfg['EnKF']['R_varname']].values)
    R = R.reshape([len(da_meas['lat']), len(da_meas['lon']), 1, 1])

# --- Calculate state perturbation covariance matrix --- #
# Calculate perturvation magnitude [nlayer, lat, lon]
da_scale = xr.open_dataset(os.path.join(
    cfg['CONTROL']['root_dir'], cfg['EnKF']['state_perturb_nc']))\
    [cfg['EnKF']['scale_varname']]
# Extract veg_class and snow_band information from a state file
if not linear_model:
    init_state_nc = os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['EnKF']['vic_initial_state'])
else:
    init_state_nc = os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['LINEAR_MODEL']['initial_state'])
ds_state = xr.open_dataset(init_state_nc)
nlayer = len(ds_state['nlayer'])
nveg = len(ds_state['veg_class'])
nsnow = len(ds_state['snow_band'])
n = nlayer * nveg * nsnow
# Calculate Cholesky L
L = calculate_cholesky_L(n, cfg['EnKF']['state_perturb_corrcoef'])
# Calculate scale for state perturbation
scale_n_nloop = calculate_scale_n_whole_field(
                    da_scale, nveg, nsnow)  # [nloop, n]
# Calculate maximum soil moisture for each tile [lat, lon, n]
da_max_moist = calculate_max_soil_moist_domain(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['VIC']['vic_global_template']))
da_max_moist_n = convert_max_moist_n_state(da_max_moist, nveg, nsnow)
# If linear model subsitution, no max moist limit
if linear_model:
    da_max_moist_n[:, :, :] = 99999

# --- For mismatched grid case, load weight information --- #
if cfg['GRID_MISMATCH']['mismatched_grid']:
    weight_nc = os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['GRID_MISMATCH']['weight_nc'])
else:
    weight_nc = None

# -------------------------------------------------------- #
# --- Run EnKF --- #
# -------------------------------------------------------- #
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])
print('Start running EnKF for ', start_time, 'to', end_time, '...')

if not linear_model:
    dict_ens_list_history_files = EnKF_VIC(
         N=cfg['EnKF']['N'],
         start_time=start_time,
         end_time=end_time,
         init_state_nc=os.path.join(cfg['CONTROL']['root_dir'],
                                    cfg['EnKF']['vic_initial_state']),
         L=L,
         scale_n_nloop=scale_n_nloop,
         da_max_moist_n=da_max_moist_n,
         R=R,
         da_meas=da_meas,
         da_meas_time_var='time',
         vic_exe=vic_exe,
         vic_global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['VIC']['vic_global_template']),
         ens_forcing_basedir=os.path.join(cfg['CONTROL']['root_dir'],
                                           cfg['FORCINGS']['ens_forcing_basedir']),
         ens_forcing_prefix=cfg['FORCINGS']['ens_forcing_prefix'],
         orig_forcing_basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['FORCINGS']['orig_forcing_nc_basepath']),
         vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
         output_vic_global_root_dir=dirs['global'],
         output_vic_state_root_dir=dirs['states'],
         output_vic_history_root_dir=dirs['history'],
         output_vic_log_root_dir=dirs['logs'],
         output_restart_log_dir=dirs['restart_log'],
         bias_correct=cfg['EnKF']['bias_correct'],
         mismatched_grid=cfg['GRID_MISMATCH']['mismatched_grid'],
         weight_nc=weight_nc,
         nproc=nproc,
         debug=debug,
         output_temp_dir=dirs['temp'],
         restart=restart)
else:
    dict_ens_list_history_files = EnKF_VIC(
         N=cfg['EnKF']['N'],
         start_time=start_time,
         end_time=end_time,
         init_state_nc=os.path.join(cfg['CONTROL']['root_dir'],
                                    cfg['LINEAR_MODEL']['initial_state']),
         L=L,
         scale_n_nloop=scale_n_nloop,
         da_max_moist_n=da_max_moist_n,
         R=R,
         da_meas=da_meas,
         da_meas_time_var='time',
         vic_exe=vic_exe,
         vic_global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['VIC']['vic_global_template']),
         ens_forcing_basedir=os.path.join(cfg['CONTROL']['root_dir'],
                                           cfg['FORCINGS']['ens_forcing_basedir']),
         ens_forcing_prefix=cfg['FORCINGS']['ens_forcing_prefix'],
         orig_forcing_basepath=cfg['FORCINGS']['orig_forcing_nc_basepath'],
         vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
         output_vic_global_root_dir=dirs['global'],
         output_vic_state_root_dir=dirs['states'],
         output_vic_history_root_dir=dirs['history'],
         output_vic_log_root_dir=dirs['logs'],
         output_restart_log_dir=dirs['restart_log'],
         bias_correct=cfg['EnKF']['bias_correct'],
         nproc=nproc,
         debug=debug,
         output_temp_dir=dirs['temp'],
         restart=restart,
         linear_model='True',
         linear_model_prec_varname=prec_varname,
         dict_linear_model_param=dict_linear_model_param)

