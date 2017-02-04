
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
# Open-loop run
# ============================================================ #
# --- Determine open-loop run period --- #
vic_run_start_time = pd.to_datetime(cfg['OPENLOOP']['start_time'])
vic_run_end_time = pd.to_datetime(cfg['OPENLOOP']['end_time'])

print('Running open-loop: ', vic_run_start_time, 'to', vic_run_end_time,
       '...')

global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])

# --- If running linear model substite instead of VIC --- #
if 'LINEAR_MODEL' in cfg:
    linear_model = True
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

# --- Run VIC (orig. forcings and states) --- #
# Identify initial state time
init_state_time = vic_run_start_time
# Prepare log sub-directory
out_log_dir = dirs['logs']
if not linear_model:
    propagate(
        start_time=vic_run_start_time, end_time=vic_run_end_time,
        vic_exe=vic_exe,
        vic_global_template_file=global_template,
        vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
        init_state_nc=os.path.join(
                             cfg['CONTROL']['root_dir'],
                             cfg['VIC']['vic_initial_state']),
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
else:
    propagate_linear_model(
        start_time=vic_run_start_time, end_time=vic_run_end_time,
        lat_coord=lat_coord, lon_coord=lon_coord,
        model_steps_per_day=cfg['VIC']['model_steps_per_day'],
        init_state_nc=os.path.join(
                             cfg['CONTROL']['root_dir'],
                             cfg['LINEAR_MODEL']['initial_state']),
        out_state_basepath=os.path.join(dirs['states'], 'state.openloop'),
        out_history_dir=dirs['history'],
        out_history_fileprefix='history.openloop',
        forcing_basepath=os.path.join(
                                 cfg['CONTROL']['root_dir'],
                                 cfg['FORCINGS']['orig_forcing_nc_basepath']),
        prec_varname=prec_varname,
        dict_linear_model_param=dict_linear_model_param)


