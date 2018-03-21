
''' This script runs VIC from initial state as openloop.

    Usage:
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
import argparse
import time

from tonic.models.vic.vic import VIC
from tonic.io import read_config, read_configobj
from da_utils import (setup_output_dirs, propagate,
                      calculate_max_soil_moist_domain)

# ============================================================ #
# Process command line arguments
# ============================================================ #
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='config file')
parser.add_argument('--mpi_proc', help='nproc for VIC run', type=int)
parser.add_argument('--state_pert_time',
                    help='Time of state perturbation, YYYY-MM-DD-HH-SS')
parser.add_argument('--sm_pert',
                    help='Perturbation amount for each layer; unit: [mm]. ' \
                         'Note: this is the deterministic amount, not a '\
                         'statistical magnitude; value can be positive or '\
                         'negative',
                    type=float,
                    nargs='+')
args = parser.parse_args()

# --- Process command line arguments
# Read config file
cfg = read_configobj(args.cfg)
# Number of processors for each VIC run
mpi_proc = args.mpi_proc
# State perturbation time
state_pert_time = pd.to_datetime(args.state_pert_time)
# State perturbation amount
sm_pert = args.sm_pert

print('sm_pert: ', sm_pert)

# ============================================================ #
# Prepare output directories
# ============================================================ #
dirs = setup_output_dirs(
    os.path.join(cfg['CONTROL']['root_dir'],
                 cfg['OUTPUT']['output_basedir']),
    mkdirs=['global', 'history', 'states', 'logs'])

# ============================================================ #
# Prepare VIC exe and MPI exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))
mpi_exe = cfg['VIC']['mpi_exe']

# ============================================================ #
# Run VIC
# ============================================================ #
# --- Extract VIC run basic information --- #
global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])
model_steps_per_day = cfg['VIC']['model_steps_per_day']

# --- Run VIC until the state perturbation time point --- #
start_time = pd.to_datetime(cfg['PERTURBATION_TEST']['start_time'])
end_time = pd.to_datetime(cfg['PERTURBATION_TEST']['end_time'])
# If state perturbation after start_time
if state_pert_time > start_time:
    # Determine running period
    vic_run_start_time = start_time
    vic_run_end_time = state_pert_time - \
                       pd.DateOffset(hours=24/model_steps_per_day)
    print('Running VIC from {} to {}'.format(vic_run_start_time, vic_run_end_time))
    # Run VIC
    init_state_nc = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['VIC']['vic_initial_state'])
    propagate(
        start_time=vic_run_start_time, end_time=vic_run_end_time,
        vic_exe=vic_exe,
        vic_global_template_file=global_template,
        vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
        init_state_nc=init_state_nc,
        out_state_basepath=os.path.join(dirs['states'], 'state.openloop'),
        out_history_dir=dirs['history'],
        out_history_fileprefix='history.openloop',
        out_global_basepath=os.path.join(dirs['global'], 'global.openloop'),
        out_log_dir=dirs['logs'],
        forcing_basepath=os.path.join(
                                 cfg['CONTROL']['root_dir'],
                                 cfg['FORCINGS']['orig_forcing_nc_basepath']),
        mpi_proc=mpi_proc,
        mpi_exe=mpi_exe)

# --- Perturb soil moisture states --- #
# Load openloop initial state
if state_pert_time > start_time:
    init_state_openloop_nc = os.path.join(
            dirs['states'],
            'state.openloop.{}_{:05d}.nc'.format(
                state_pert_time.strftime('%Y%m%d'),
                state_pert_time.hour*3600+state_pert_time.second))
else:
    init_state_openloop_nc = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['VIC']['vic_initial_state'])
time.sleep(int(np.random.uniform(0, 30)))  # This is to prevent multiple processes trying to open the same file when parallelized
ds_init_state_openloop = xr.open_dataset(init_state_openloop_nc)
# Calculate maximum soil moisture for each tile [nlyaer, lat, lon]
da_max_moist = calculate_max_soil_moist_domain(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['VIC']['vic_global_template']))
# Perturb states
ds_init_state_perturbed = ds_init_state_openloop.copy()
for l in range(len(sm_pert)):
    ds_init_state_perturbed['STATE_SOIL_MOISTURE'][:, :, l, :, :] += sm_pert[l]
    # Limit perturbation to be between zero and upper bound
    for lat in da_max_moist['lat'].values:
        for lon in da_max_moist['lon'].values:
            sm = ds_init_state_perturbed['STATE_SOIL_MOISTURE']\
                    .loc[:, :, l, lat, lon].values
            # Set negative to zero
            sm[sm<0] = 0
            # Set above-maximum to maximum
            max_moist = da_max_moist.sel(lat=lat, lon=lon, nlayer=l).values
            sm[sm>max_moist] = max_moist
            # Put back into state ds
            ds_init_state_perturbed['STATE_SOIL_MOISTURE']\
                .loc[:, :, l, lat, lon] = sm
# Save perturbed states to netCDF file
perturbed_state_nc = os.path.join(
    dirs['states'], 'state.perturbed.{}_{}_{}.{}_{:05d}.nc'.format(
        sm_pert[0], sm_pert[1], sm_pert[2],
        state_pert_time.strftime('%Y%m%d'),
        state_pert_time.hour*3600+state_pert_time.second))
ds_init_state_perturbed.to_netcdf(perturbed_state_nc)

# --- Run VIC from perturbed states to end_time --- #
# Determine running period
vic_run_start_time = state_pert_time
vic_run_end_time = end_time
print('Running VIC from {} to {}'.format(vic_run_start_time, vic_run_end_time))
# Run VIC
history_fileprefix = 'history.perturbed_init_state.{}_{}_{}'.format(
    sm_pert[0], sm_pert[1], sm_pert[2])
out_global_basepath = os.path.join(
    dirs['global'],
    'global.perturbed_init_state.{}_{}_{}'.format(
        sm_pert[0], sm_pert[1], sm_pert[2]))
propagate(
    start_time=vic_run_start_time, end_time=vic_run_end_time,
    vic_exe=vic_exe,
    vic_global_template_file=global_template,
    vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
    init_state_nc=perturbed_state_nc,
    out_state_basepath=None,
    out_history_dir=dirs['history'],
    out_history_fileprefix=history_fileprefix,
    out_global_basepath=out_global_basepath,
    out_log_dir=dirs['logs'],
    forcing_basepath=os.path.join(
                             cfg['CONTROL']['root_dir'],
                             cfg['FORCINGS']['orig_forcing_nc_basepath']),
    mpi_proc=mpi_proc,
    mpi_exe=mpi_exe)


