# =========================================================== #
# This script produces true and synthetic surface soil moisture measurements
#    - Run VIC with "truth" forcings and perturbed states --> "truth"
#    - Add random noise to "truth" top-layer soil moisture --> synthetic measurements
# =========================================================== #

import sys
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from tonic.models.vic.vic import VIC
from tonic.io import read_configobj

from da_utils import (Forcings, setup_output_dirs, propagate,
                      calculate_sm_noise_to_add_magnitude,
                      perturb_soil_moisture_states,
                      calculate_max_soil_moist_domain,
                      convert_max_moist_n_state, VarToPerturb,
                      find_global_param_value, propagate_linear_model,
                      concat_clean_up_history_file,
                      calculate_scale_n_whole_field,
                      calculate_cholesky_L)

# =========================================================== #
# Load command line arguments
# =========================================================== #
cfg = read_configobj(sys.argv[1])
mpi_proc = int(sys.argv[2])

# =========================================================== #
# Set random generation seed
# =========================================================== #
np.random.seed(cfg['CONTROL']['seed'])

# =========================================================== #
# Process some config parameters
# =========================================================== #
print('Processing config parameters...')
# Simulation time
start_time = pd.to_datetime(cfg['TIME_INDEX']['start_time'])
end_time = pd.to_datetime(cfg['TIME_INDEX']['end_time'])

# Set up output sub-directories
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_basedir']),
                         mkdirs=['truth', 'synthetic_meas', 'plots'])
truth_subdirs = setup_output_dirs(dirs['truth'],
                                  mkdirs=['global', 'history', 'states',
                                          'logs'])

# Construct time points for synthetic measurement (daily, at a certain hour)
# (1) Determine first and last measurement time point
if start_time.hour >= cfg['TIME_INDEX']['synthetic_meas_hour']:
    next_day = start_time + pd.DateOffset(days=1)
    meas_start_time = pd.datetime(next_day.year, next_day.month, next_day.day,
                                  cfg['TIME_INDEX']['synthetic_meas_hour'])
else:
    meas_start_time = pd.datetime(start_time.year, start_time.month, start_time.day,
                                  cfg['TIME_INDEX']['synthetic_meas_hour'])
if end_time.hour <= cfg['TIME_INDEX']['synthetic_meas_hour']:
    last_day = end_time - pd.DateOffset(days=1)
    meas_end_time = pd.datetime(last_day.year, last_day.month, last_day.day,
                                cfg['TIME_INDEX']['synthetic_meas_hour'])
else:
    meas_end_time = pd.datetime(end_time.year, end_time.month, end_time.day,
                                cfg['TIME_INDEX']['synthetic_meas_hour'])
# (2) Construct measurement time series
meas_times = pd.date_range(meas_start_time, meas_end_time, freq='D')

# VIC global template file
global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])

# Process linear model subsitute, if specified
if 'LINEAR_MODEL' in cfg:
    linear_model = True
    adjust_negative = False
    prec_varname = cfg['LINEAR_MODEL']['prec_varname']
    dict_linear_model_param={'r1': cfg['LINEAR_MODEL']['r1'],
                             'r2': cfg['LINEAR_MODEL']['r2'],
                             'r3': cfg['LINEAR_MODEL']['r3'],
                             'r12': cfg['LINEAR_MODEL']['r12'],
                             'r23': cfg['LINEAR_MODEL']['r23']}
    # Find VIC domain file and extract lat & lon
    with open(global_template, 'r') as global_file:
        gp = global_file.read()
    domain_nc = find_global_param_value(gp, 'DOMAIN')
    ds_nc = xr.open_dataset(domain_nc)
    lat_coord = ds_nc['lat']
    lon_coord = ds_nc['lon']
else:
    linear_model = False
    adjust_negative = True

# =========================================================== #
# Simulate "truth" - run VIC with perturbed forcings and
# states
# =========================================================== #
print('Simulating \"truth\" - run VIC with "true" forcings and perturbed states...')
# --- Create class VIC --- #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['exe']))

# --- Run VIC with "truth" forcings and perturbed soil moisture states --- #
# Initialize a list of file paths to be concatenated
list_history_paths = []

# (1) Run VIC until the first measurement time point (with initial state)
run_end_time = meas_times[0] - pd.DateOffset(days=1/cfg['VIC']['model_steps_per_day'])
prop_period_stamp = '{}-{}'.format(start_time.strftime('%Y%m%d_%H%S'),
                                   run_end_time.strftime('%Y%m%d_%H%S'))
print('\tRun VIC until the first measurement time {}...'.format(prop_period_stamp))
# Prepare log directories
log_dir = setup_output_dirs(
                    truth_subdirs['logs'],
                    mkdirs=['propagate.{}'.format(prop_period_stamp)])\
          ['propagate.{}'.format(prop_period_stamp)]
# Propagate until the first measurement point
if not linear_model:
    propagate(start_time=start_time, end_time=run_end_time,
              vic_exe=vic_exe, vic_global_template_file=global_template,
              vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
              init_state_nc=os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['VIC']['vic_initial_state']),
              out_state_basepath=os.path.join(truth_subdirs['states'],
                                              'propagated.state'),
              out_history_dir=truth_subdirs['history'],
              out_history_fileprefix='history',
              out_global_basepath=os.path.join(truth_subdirs['global'], 'global'),
              out_log_dir=log_dir,
              forcing_basepath=os.path.join(
                      cfg['CONTROL']['root_dir'],
                      cfg['VIC']['truth_forcing_nc_basepath']),
              mpi_proc=mpi_proc,
              mpi_exe=cfg['VIC']['mpi_exe'])
else:
    propagate_linear_model(
            start_time=start_time,
            end_time=run_end_time,
            lat_coord=lat_coord,
            lon_coord=lon_coord,
            model_steps_per_day=cfg['VIC']['model_steps_per_day'],
            init_state_nc=os.path.join(cfg['CONTROL']['root_dir'],
                                       cfg['LINEAR_MODEL']['initial_state']),
            out_state_basepath=os.path.join(truth_subdirs['states'],
                                            'propagated.state'),
            out_history_dir=truth_subdirs['history'],
            out_history_fileprefix='history',
            forcing_basepath=os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['VIC']['truth_forcing_nc_basepath']),
            prec_varname=prec_varname,
            dict_linear_model_param=dict_linear_model_param)

# Concat output history file to the list to be concatenated
list_history_paths.append(os.path.join(truth_subdirs['history'],
                                       'history.{}-{:05d}.nc'.format(
                                            start_time.strftime('%Y-%m-%d'),
                                            start_time.hour*3600+start_time.second)))

# (2) Loop over until each measurement point and run VIC
# --- Calculate state perturbation covariance matrix --- #
# Calculate perturvation magnitude [nlayer, lat, lon]
da_scale = calculate_sm_noise_to_add_magnitude(
                vic_history_path=os.path.join(
                        cfg['CONTROL']['root_dir'],
                        cfg['FORCINGS_STATES_PERTURB']['vic_history_path']),
                sigma_percent=cfg['FORCINGS_STATES_PERTURB']['state_perturb_sigma_percent'])
# Extract veg_class and snow_band information from a state file
state_time = meas_times[0]
state_filename = os.path.join(truth_subdirs['states'],
                              'propagated.state.{}_{:05d}.nc'.format(
                                    state_time.strftime('%Y%m%d'),
                                    state_time.hour*3600+state_time.second))
ds_state = xr.open_dataset(state_filename)
nlayer = len(ds_state['nlayer'])
nveg = len(ds_state['veg_class'])
nsnow= len(ds_state['snow_band'])
n = nlayer * nveg * nsnow
# Calculate Cholesky L
L = calculate_cholesky_L(n, cfg['FORCINGS_STATES_PERTURB']['state_perturb_corrcoef'])
# Calculate scale for state perturbation
scale_n_nloop = calculate_scale_n_whole_field(
                    da_scale, nveg, nsnow)  # [nloop, n]
# Calculate maximum soil moisture for each tile [lat, lon, n]
da_max_moist = calculate_max_soil_moist_domain(global_template)
da_max_moist_n = convert_max_moist_n_state(da_max_moist, nveg, nsnow)
# If linear model subsitution, no max moist limit
if linear_model:
    da_max_moist_n[:, :, :] = 99999

# --- Run VIC --- #
for t in range(len(meas_times)):
    # --- Determine current and next time point (all these are time
    # points at the beginning of a time step)--- #
    current_time = meas_times[t]
    if t == len(meas_times)-1:  # if this is the last measurement time
        next_time = end_time
    else:  # if not the last measurement time
        next_time = meas_times[t+1] - pd.DateOffset(days=1/cfg['VIC']['model_steps_per_day'])
    # If current_time > next_time, do nothing (we already reach the end of the simulation)
    if current_time > next_time:
        break
    print('\tRun VIC ', current_time, 'to', next_time, '("true" forcings and perturbed states)')

    # --- Perturb states --- #
    state_time = current_time
    orig_state_nc = os.path.join(
                            truth_subdirs['states'],
                            'propagated.state.{}_{:05d}.nc'.format(
                                    state_time.strftime('%Y%m%d'),
                                    state_time.hour*3600+state_time.second))
    perturbed_state_nc = os.path.join(
                            truth_subdirs['states'],
                            'perturbed.state.{}_{:05d}.nc'.format(
                                    state_time.strftime('%Y%m%d'),
                                    state_time.hour*3600+state_time.second))
    da_perturbation = perturb_soil_moisture_states(
            states_to_perturb_nc=orig_state_nc,
            L=L,
            scale_n_nloop=scale_n_nloop,
            out_states_nc=perturbed_state_nc,
            da_max_moist_n=da_max_moist_n,
            adjust_negative=adjust_negative,
            seed=None)
    # Clean up original state file
    os.remove(orig_state_nc)

    # --- Propagate to the next time point --- #
    # Prepare log directories
    prop_period_stamp = '{}-{}'.format(current_time.strftime('%Y%m%d_%H%S'),
                                       next_time.strftime('%Y%m%d_%H%S'))
    log_dir = setup_output_dirs(
                    truth_subdirs['logs'],
                    mkdirs=['propagate.{}'.format(prop_period_stamp)])\
              ['propagate.{}'.format(prop_period_stamp)]
    if not linear_model:
        propagate(
            start_time=current_time, end_time=next_time,
            vic_exe=vic_exe, vic_global_template_file=global_template,
            vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
            init_state_nc=perturbed_state_nc,
            out_state_basepath=os.path.join(truth_subdirs['states'],
                                            'propagated.state'),
            out_history_dir=truth_subdirs['history'],
            out_history_fileprefix='history',
            out_global_basepath=os.path.join(truth_subdirs['global'], 'global'),
            out_log_dir=log_dir,
            forcing_basepath=os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['VIC']['truth_forcing_nc_basepath']),
            mpi_proc=mpi_proc,
            mpi_exe=cfg['VIC']['mpi_exe'])
    else:
        propagate_linear_model(
            start_time=current_time, end_time=next_time,
            lat_coord=lat_coord,
            lon_coord=lon_coord,
            model_steps_per_day=cfg['VIC']['model_steps_per_day'],
            init_state_nc=perturbed_state_nc,
            out_state_basepath=os.path.join(truth_subdirs['states'],
                                            'propagated.state'),
            out_history_dir=truth_subdirs['history'],
            out_history_fileprefix='history',
            forcing_basepath=os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['VIC']['truth_forcing_nc_basepath']),
            prec_varname=prec_varname,
            dict_linear_model_param=dict_linear_model_param)
    # Concat output history file to the list to be concatenated
    list_history_paths.append(os.path.join(truth_subdirs['history'],
                                           'history.{}-{:05d}.nc'.format(
                                                current_time.strftime('%Y-%m-%d'),
                                                current_time.hour*3600+current_time.second)))
    ds = xr.open_dataset(os.path.join(truth_subdirs['history'],
                                           'history.{}-{:05d}.nc'.format(
                                                current_time.strftime('%Y-%m-%d'),
                                                current_time.hour*3600+current_time.second)))
    # Clean up perturbed state file
    os.remove(perturbed_state_nc)

# (3) Concatenate all history files
hist_concat_nc = os.path.join(truth_subdirs['history'],
                                 'history.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                        start_time.strftime('%Y%m%d'),
                                        start_time.hour*3600+start_time.second,
                                        end_time.strftime('%Y%m%d'),
                                        end_time.hour*3600+end_time.second))
concat_clean_up_history_file(list_history_paths,
                             hist_concat_nc)


# =========================================================== #
# Simulate synthetic measurement - Extract top-layer soil
# moisture from "truth" at the end of each day, and add noise
# =========================================================== #

print('Simulating synthetic measurements...')

# --- Load history file --- #
ds_hist = xr.open_dataset(hist_concat_nc)

# --- Select out times of measurement --- #
ds_hist_meas_times = ds_hist.sel(time=meas_times)

# --- Select top-layer soil moisture --- #
da_sm1_true = ds_hist_meas_times['OUT_SOIL_MOIST'].sel(nlayer=0)

# --- Add noise --- #
# Generate the standard deviation of noise to be added for each grid cell
da_sigma = da_sm1_true[0, :, :].copy(deep=True)
da_sigma[:] = cfg['SYNTHETIC_MEAS']['sigma']
# Add noise
VarToPerturb_sm1 = VarToPerturb(da_sm1_true) # create class
da_sm1_perturbed = VarToPerturb_sm1.add_gaussian_white_noise(
                        da_sigma, da_max_moist.sel(nlayer=0),
                        adjust_negative)

# --- Save synthetic measurement to netCDF file --- #
ds_simulated = xr.Dataset({'simulated_surface_sm': da_sm1_perturbed})
ds_simulated.to_netcdf(os.path.join(dirs['synthetic_meas'],
                                    'synthetic_meas.{}_{}.nc'.format(start_time.strftime('%Y%m%d'),
                                                                     end_time.strftime('%Y%m%d'))),
                       format='NETCDF4_CLASSIC')

