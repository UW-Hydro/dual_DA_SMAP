
''' This script runs VIC from initial state as openloop. Save state files at
    time of measurements.

    Usage:
        $ python run_data_assim.py <config_file_EnKF> mpi_proc
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

from tonic.models.vic.vic import VIC
from tonic.io import read_config, read_configobj
from da_utils import (setup_output_dirs, propagate, find_global_param_value,
                      propagate_linear_model)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Number of processors for each VIC run
mpi_proc = int(sys.argv[2])


# ============================================================ #
# Prepare output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_openloop_basedir']),
                         mkdirs=['global', 'history', 'states',
                                 'logs', 'plots'])


# ============================================================ #
# Prepare VIC exe and MPI exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))
mpi_exe = cfg['VIC']['mpi_exe']


# ============================================================ #
# Construct time points at measurement times
# ============================================================ #
start_time = pd.to_datetime(cfg['OPENLOOP']['start_time'])
end_time = pd.to_datetime(cfg['OPENLOOP']['end_time'])

# Construct time points for synthetic measurement (daily, at a certain hour)
# (1) Determine first and last measurement time point
if start_time.hour >= cfg['OPENLOOP']['synthetic_meas_hour']:
    next_day = start_time + pd.DateOffset(days=1)
    meas_start_time = pd.datetime(next_day.year, next_day.month, next_day.day,
                                  cfg['OPENLOOP']['synthetic_meas_hour'])
else:
    meas_start_time = pd.datetime(start_time.year, start_time.month, start_time.day,
                                  cfg['OPENLOOP']['synthetic_meas_hour'])
if end_time.hour <= cfg['OPENLOOP']['synthetic_meas_hour']:
    last_day = end_time - pd.DateOffset(days=1)
    meas_end_time = pd.datetime(last_day.year, last_day.month, last_day.day,
                                cfg['OPENLOOP']['synthetic_meas_hour'])
else:
    meas_end_time = pd.datetime(end_time.year, end_time.month, end_time.day,
                                cfg['OPENLOOP']['synthetic_meas_hour'])
# (2) Construct measurement time series
meas_times = pd.date_range(meas_start_time, meas_end_time, freq='D')


# ============================================================ #
# Open-loop run, with states saved
# ============================================================ #
# --- Prepare VIC run setup --- #
global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])
out_log_dir = dirs['logs']
forcing_basepath = os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['FORCINGS']['orig_forcing_nc_basepath'])
vic_model_steps_per_day = cfg['VIC']['model_steps_per_day']

# --- Run until the first measurement time point --- #
vic_run_start_time = start_time
vic_run_end_time = meas_times[0] - \
                   pd.DateOffset(hours=24/vic_model_steps_per_day)
print('Running: ', vic_run_start_time, 'to', vic_run_end_time,
       '...')
# Prepare parameters
init_state_nc = os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['VIC']['vic_initial_state'])
# Run VIC
propagate(
    start_time=vic_run_start_time, end_time=vic_run_end_time,
    vic_exe=vic_exe,
    vic_global_template_file=global_template,
    vic_model_steps_per_day=vic_model_steps_per_day,
    init_state_nc=init_state_nc,
    out_state_basepath=os.path.join(dirs['states'], 'state.openloop'),
    out_history_dir=dirs['history'],
    out_history_fileprefix='history.openloop',
    out_global_basepath=os.path.join(dirs['global'], 'global.openloop'),
    out_log_dir=out_log_dir,
    forcing_basepath=forcing_basepath,
    mpi_proc=mpi_proc,
    mpi_exe=mpi_exe)

# --- Loop over each measurement point and run VIC --- #
for t, time in enumerate(meas_times):
    # --- Determine last, current and next measurement time points --- #
    current_time = time
    if t == len(meas_times)-1:  # if this is the last measurement time
        next_time = end_time
    else:  # if not the last measurement time
        next_time = meas_times[t+1] - \
                    pd.DateOffset(hours=24/vic_model_steps_per_day)
    if current_time > next_time:
            break
    print('Running:', current_time, 'to', next_time, '...')
    # --- Run VIC --- #
    # Prepare parameters
    init_state_nc = os.path.join(
        dirs['states'],
        'state.openloop.{}_{:05d}.nc'.format(
            current_time.strftime('%Y%m%d'),
            current_time.hour*3600+current_time.second))
    # Run VIC
    propagate(
        start_time=current_time, end_time=next_time,
        vic_exe=vic_exe,
        vic_global_template_file=global_template,
        vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
        init_state_nc=init_state_nc,
        out_state_basepath=os.path.join(dirs['states'], 'state.openloop'),
        out_history_dir=dirs['history'],
        out_history_fileprefix='history.openloop',
        out_global_basepath=os.path.join(dirs['global'], 'global.openloop'),
        out_log_dir=out_log_dir,
        forcing_basepath=forcing_basepath,
        mpi_proc=mpi_proc,
        mpi_exe=mpi_exe)

