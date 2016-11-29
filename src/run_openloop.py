
''' This script runs VIC from initial state as openloop.

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
from da_utils import setup_output_dirs, propagate


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Number of processors for each VIC run
mpi_proc = int(sys.argv[2])

# ============================================================ #
# Set random generation seed
# ============================================================ #
np.random.seed(cfg['CONTROL']['seed'])


# ============================================================ #
# Prepare output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_EnKF_basedir']),
                         mkdirs=['global', 'history', 'states',
                                 'logs', 'plots'])


# ============================================================ #
# Prepare VIC exe and MPI exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))
mpi_exe = cfg['VIC']['mpi_exe']


# ============================================================ #
# Open-loop run
# ============================================================ #
# --- Determine open-loop run period --- #
vic_run_start_time = pd.to_datetime(cfg['EnKF']['start_time'])
vic_run_end_time = pd.to_datetime(cfg['EnKF']['end_time'])

print('Running open-loop: ', vic_run_start_time, 'to', vic_run_end_time,
       '...')

# --- Run VIC (orig. forcings and states) --- #
# Identify initial state time
init_state_time = vic_run_start_time
# Prepare log sub-directory
out_log_dir = setup_output_dirs(dirs['logs'], mkdirs=['openloop'])['openloop']
propagate(start_time=vic_run_start_time, end_time=vic_run_end_time,
          vic_exe=vic_exe,
          vic_global_template_file=os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['VIC']['vic_global_template']),
           vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
           init_state_nc=os.path.join(
                                cfg['CONTROL']['root_dir'],
                                cfg['EnKF']['vic_initial_state']),
           out_state_basepath=os.path.join(dirs['states'], 'state.openloop'),
           out_history_dir=dirs['history'],
           out_history_fileprefix='history.openloop',
           out_global_basepath=os.path.join(dirs['global'], 'global.openloop'),
           out_log_dir=out_log_dir,
           forcing_basepath=os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['FORCINGS']['orig_forcing_nc_basepath']),
           mpi_proc=mpi_proc,
           mpi_exe=mpi_exe)
hist_openloop_nc = os.path.join(
                        dirs['history'],
                        'history.openloop.{}-{:05d}.nc'.format(
                                vic_run_start_time.strftime('%Y-%m-%d'),
                                vic_run_start_time.hour*3600+vic_run_start_time.second))


