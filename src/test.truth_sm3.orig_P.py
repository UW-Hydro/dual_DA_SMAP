
''' This script run VIC with "truth" sm1 and "openloop" lower layer sm states,
    and original forcing.

    Usage:
        $ python test.truth_sm3.orig_P.py <gen_synth_config_file> <mpi_proc>
'''

from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import xarray as xr
import sys
import multiprocessing as mp
import shutil
import subprocess

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic import VIC
from da_utils import (setup_output_dirs, run_vic_assigned_states,
                      concat_vic_history_files,
                      calculate_ensemble_mean_states,
                      Forcings, to_netcdf_state_file_compress)

# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Read number of processors for VIC MPI runs
mpi_proc = int(sys.argv[2])

# ============================================================ #
# Prepare VIC exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['exe']))


# ============================================================ #
# Process cfg data
# ============================================================ #
start_time = pd.to_datetime(cfg['TIME_INDEX']['start_time'])
end_time = pd.to_datetime(cfg['TIME_INDEX']['end_time'])

start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Setup postprocess output directories
# ============================================================ #
basedir = setup_output_dirs(
    os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_basedir']),
    mkdirs=['test.truth_sm3_orig_forcing'])['test.truth_sm3_orig_forcing']
dirs = setup_output_dirs(basedir,
                         mkdirs=['global', 'states', 'history', 'logs'])


# ============================================================ #
# Load data
# ============================================================ #
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


# ============================================================ #
# Run VIC with "truth" sm3 states and original forcings
# ============================================================ #
# --- Generate state files with "truth" sm3 and openloop other states --- #
print('Generating antecedent state files...')
dict_assigned_state_nc = OrderedDict()
for t, time in enumerate(meas_times):
    print(time)
    # Load "truth" state
    ds_state_truth = xr.open_dataset(os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_basedir'],
        'truth',
        'states',
        'perturbed.state.{}_{:05d}.nc'.format(
                time.strftime('%Y%m%d'),
                time.hour*3600+time.second)))
    # Load "openloop" state
    ds_state_openloop = xr.open_dataset(os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OPENLOOP']['openloop_basedir'],
        'states',
        'state.openloop.{}_{:05d}.nc'.format(
                time.strftime('%Y%m%d'),
                time.hour*3600+time.second)))
    # Replace sm3 with truth
    ds_state_openloop['STATE_SOIL_MOISTURE'].loc[:, :, 2, :, :] = \
        ds_state_truth['STATE_SOIL_MOISTURE'].loc[:, :, 2, :, :]
    # Save new state to netCDF file
    out_nc = os.path.join(
        dirs['states'],
        'state.{}_{:05d}.nc'.format(
                time.strftime('%Y%m%d'),
                time.hour*3600+time.second))
    to_netcdf_state_file_compress(ds_state_openloop, out_nc)
    # Put new state nc to dictionary
    dict_assigned_state_nc[time] = out_nc

# --- Run VIC --- #
# Prepare some variables
init_state_nc = os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['VIC']['vic_initial_state'])
# other variables
global_template = os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['VIC']['vic_global_template'])
vic_forcing_basepath = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['VIC']['orig_forcing_nc_basepath'])
# --- run VIC with assigned states --- #
print('Run VIC with "truth" sm3 states and orig. forcing...')
list_history_files = run_vic_assigned_states(
    start_time=start_time,
    end_time=end_time,
    vic_exe=vic_exe,
    init_state_nc=init_state_nc,
    dict_assigned_state_nc=dict_assigned_state_nc,
    global_template=global_template,
    vic_forcing_basepath=vic_forcing_basepath,
    vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
    output_global_root_dir=dirs['global'],
    output_vic_history_root_dir=dirs['history'],
    output_vic_log_root_dir=dirs['logs'],
    mpi_proc=mpi_proc,
    mpi_exe=cfg['VIC']['mpi_exe'])

# --- Concat by-year history files --- #
print('Concatenating by-year history files...')
subprocess.call(
    "cdo copy {} {}".format(
        os.path.join(dirs['history'], '*'),
        os.path.join(dirs['history'], 'history.concat.{}_{}.nc'.format(
            start_time.strftime('%Y%m%d'), end_time.strftime('%Y%m%d')))),
    shell=True)
# Clean up
for year in range(start_year, end_year+1):
    os.remove(os.path.join(dirs['history'],
                           'history.concat.{}.nc'.format(year)))
