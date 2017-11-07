import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic import VIC

from da_utils import (Forcings, perturb_forcings_ensemble, setup_output_dirs,
                      to_netcdf_forcing_file_compress, calculate_sm_noise_to_add_magnitude,
                      calculate_scale_n_whole_field, to_netcdf_state_file_compress,
                      run_vic_assigned_states)

# ===================================================== #
# Parameter setting
# ===================================================== #
lat = 34.6875
lon = -94.9375

# Root directory - all other paths will be under root_dir
root_dir = '/civil/hydro/ymao/data_assim/'

# --- Time --- #
start_time = pd.to_datetime('1980-01-01-00')
end_time = pd.to_datetime('1989-12-31-21')
start_year = start_time.year
end_year = end_time.year
state_times = pd.date_range(start_time, end_time, freq='D')

# --- Inputs --- #
# Orig. forcing netcdf basepath ('YYYY.nc' will be appended)
force_orig_nc = os.path.join(root_dir, 'forcing/vic/Newman/{}_{}/ens_100/force.'.format(lat, lon))
# Orig. history netcdf file
hist_orig_nc = os.path.join(
    root_dir,
    ('output/vic/ArkRed/openloop.1980_1989.Maurer_param/'
     'history/history.openloop.{}_{}.1980-01-01-00000.nc').format(lat, lon))
# Orig state basepath ('YYYYMMDD_SSSSS.nc' will be appended); initial state not included
state_orig_basepath = os.path.join(
    root_dir,
    'output/vic/ArkRed/openloop.1980_1989.Maurer_param/states/{}_{}/state.openloop.'.format(lat, lon))
# Initial state file
init_state_nc = os.path.join(
    root_dir,
    'output/vic/ArkRed/spinup.1949_1979.Maurer_param/states/state.{}_{}.19800101_00000.nc'.format(lat, lon))

# --- Perturbation --- #
### prec. perturbation ###
# Number of prec. perturbation ensemble
N_prec = 3
# Standard deviation of prec. perturbation multiplier (fixed)
prec_std = 1
# Parameter in AR(1) process for prec. perturbation (fixed)
# prec_phi = 0
### State perturbation ###
# Number of state perturbation ensemble
N_state = 3
# Percentage of max value of each state to perturb (e.g., if
# state_perturb_sigma_percent = 5, then Gaussian noise with standard deviation
# = 5% of max soil moisture will be added as perturbation)
# state_perturb_sigma_percent is a list of percentage for each soil layer
# (e.g., if there are three soil layers, then an example of state_perturb_sigma_percent is: 20,1,0.5)
# NOTE: keep this consistent as in the dual correction system
state_perturb_sigma_percent = [5, 5, 0.5]

# --- Run VIC --- #
# Ensemble indices of the perturbed forcing and state ensemble used (starting from 0)
ens_force = int(sys.argv[1])
ens_state = int(sys.argv[2])
# VIC exe path
vic_exe_path = os.path.join(root_dir, 'VIC/vic/drivers/image/vic_image.exe')
# VIC global template file
vic_global_template_path = os.path.join(
    root_dir, 'control/vic/hyak.global.34.6875_-94.9375.template.Maurer_param.txt')
# VIC time step
vic_model_steps_per_day = 8

# --- Outputs --- #
output_dir = os.path.join(root_dir, 'tools/error_correlation_synthetic/output/{}_{}'.format(lat, lon))

# ===================================================== #
# Set up variables needed to run VIC
# ===================================================== #
# Prepare VIC exe
vic_exe = VIC(vic_exe_path)
# State files
dict_assigned_state_nc = OrderedDict()
for time in state_times[1:]:
    dict_assigned_state_nc[time] = os.path.join(
        output_dir, 'perturbed_states', 'corrcoef_0', 'ens_{}'.format(ens_state+1),
        'state.{}_{:05d}.nc'.format(time.strftime('%Y%m%d'), time.second))
# Forcing basepath
vic_forcing_basepath = os.path.join(
    output_dir, 'perturbed_forcings', 'corrcoef_0', 'ens_{}'.format(ens_force+1), 'force.')
# Output directories
vic_output_dir = setup_output_dirs(
    output_dir, mkdirs=['vic_output'])['vic_output']
subdir_name = 'force{}_state{}'.format(ens_force+1, ens_state+1)
global_subdir = setup_output_dirs(
    vic_output_dir,
    mkdirs=['global'])['global']
global_subdir = setup_output_dirs(
    global_subdir,
    mkdirs=[subdir_name])[subdir_name]
hist_subdir = setup_output_dirs(
    vic_output_dir,
    mkdirs=['history'])['history']
hist_subdir = setup_output_dirs(
    hist_subdir,
    mkdirs=[subdir_name])[subdir_name]
log_subdir = setup_output_dirs(
    vic_output_dir,
    mkdirs=['logs'])['logs']
log_subdir = setup_output_dirs(
    log_subdir,
    mkdirs=[subdir_name])[subdir_name]

# ===================================================== #
# Run VIC with assigned states and forcing
# ===================================================== #
list_history_files = run_vic_assigned_states(
            start_time=start_time,
            end_time=end_time,
            vic_exe=vic_exe,
            init_state_nc=init_state_nc,
            dict_assigned_state_nc=dict_assigned_state_nc,
            global_template=vic_global_template_path,
            vic_forcing_basepath=vic_forcing_basepath,
            vic_model_steps_per_day=vic_model_steps_per_day,
            output_global_root_dir=global_subdir,
            output_vic_history_root_dir=hist_subdir,
            output_vic_log_root_dir=log_subdir)


