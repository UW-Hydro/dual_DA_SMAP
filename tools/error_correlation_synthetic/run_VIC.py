import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import sys
import argparse

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic import VIC

from da_utils import (Forcings, perturb_forcings_ensemble, setup_output_dirs,
                      to_netcdf_forcing_file_compress, calculate_sm_noise_to_add_magnitude,
                      calculate_scale_n_whole_field, to_netcdf_state_file_compress,
                      run_vic_assigned_states, concat_clean_up_history_file)

# ============================================================ #
# Process command line arguments
# ============================================================ #
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
                    help='Config file')
parser.add_argument('--corrcoef', type=float,
                    help='Correlation coefficient of perturbation')
parser.add_argument('--ens', type=int,
                    help='Ensemble index, starting from 1')
args = parser.parse_args()

# Read config file
cfg = read_configobj(args.cfg)
# Correlation coefficient of perturbation
corrcoef = args.corrcoef
# Perturbation ensemble size
ens = args.ens


# ===================================================== #
# Parameter setting
# ===================================================== #

# Root directory - all other paths will be under root_dir
root_dir = cfg['CONTROL']['root_dir']

# --- Time --- #
start_time = pd.to_datetime(cfg['TIME']['start_time'])
end_time = pd.to_datetime(cfg['TIME']['end_time'])
start_year = start_time.year
end_year = end_time.year
state_times = pd.date_range(start_time, end_time, freq='D')

# --- Run VIC --- #
# VIC exe path
vic_exe_path = os.path.join(root_dir, cfg['VIC']['vic_exe'])
# VIC global template file
vic_global_template_path = os.path.join(
    root_dir, cfg['INPUTS']['vic_global_template_path'])
# VIC time step
vic_model_steps_per_day = cfg['VIC']['vic_model_steps_per_day']

# --- Outputs --- #
output_dir = os.path.join(root_dir, cfg['OUTPUT']['output_dir'])


# ===================================================== #
# Set up variables needed to run VIC
# ===================================================== #
# Prepare VIC exe
vic_exe = VIC(vic_exe_path)
# State files
dict_assigned_state_nc = OrderedDict()
for i, time in enumerate(state_times):
    state_path = os.path.join(
        output_dir, 'perturbed_states', 'corrcoef_{}'.format(corrcoef),
        'ens_{}'.format(ens),
        'state.{}_{:05d}.nc'.format(time.strftime('%Y%m%d'), time.second))
    if i == 0:
        init_state_nc = state_path
    else:
        dict_assigned_state_nc[time] = state_path
# Forcing basepath
vic_forcing_basepath = os.path.join(
    output_dir, 'perturbed_forcings', 'corrcoef_{}'.format(corrcoef),
    'ens_{}'.format(ens), 'force.')
# Output directories
vic_output_dir = setup_output_dirs(
    output_dir, mkdirs=['vic_output'])['vic_output']
vic_output_dir = setup_output_dirs(
    vic_output_dir, mkdirs=['corrcoef_{}'.format(corrcoef)])['corrcoef_{}'.format(corrcoef)]

subdir_name = 'ens_{}'.format(ens)
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
run_vic_assigned_states(
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
# Concat all years
list_history_files = [os.path.join(
    hist_subdir, 'history.concat.{}.nc'.format(year))
    for year in range(start_year, end_year+1)]
hist_concat_nc = os.path.join(
    hist_subdir, 'history.concat.{}_{}.nc'.format(start_year, end_year))
concat_clean_up_history_file(list_history_files,
                             hist_concat_nc)


