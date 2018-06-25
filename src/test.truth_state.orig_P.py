
''' This script run VIC with "truth" soil moisture states and original forcing.

    Usage:
        $ python postprocess_EnKF.py <gen_synth_config_file> <mpi_proc>
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
                      Forcings)

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
    mkdirs=['test.truth_states_orig_forcing'])['test.truth_states_orig_forcing']
dirs = setup_output_dirs(basedir,
                         mkdirs=['global', 'history', 'logs'])


# ============================================================ #
# Load data
# ============================================================ #
# --- Load measurement data (to get time points) --- #
ds_meas_orig = xr.open_dataset(os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['OUTPUT']['output_basedir'],
    'synthetic_meas',
    'synthetic_meas.{}_{}.nc'.format(start_time.strftime('%Y%m%d'),
                                     end_time.strftime('%Y%m%d'))))
da_meas_orig = ds_meas_orig['simulated_surface_sm']
# Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(start_time, end_time))
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])


# ============================================================ #
# Run VIC with "truth" states and original forcings
# ============================================================ #
print('Run VIC with "truth" states and orig. forcing...')
# --- Prepare some variables --- #
# initial state nc
init_state_nc = os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['VIC']['vic_initial_state'])
# state file dict
dict_assigned_state_nc = OrderedDict()
for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
    state_time = pd.to_datetime(time)
    dict_assigned_state_nc[state_time] = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_basedir'],
        'truth',
        'states',
        'perturbed.state.{}_{:05d}.nc'.format(
                time.strftime('%Y%m%d'),
                time.hour*3600+time.second))
# other variables
global_template = os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['VIC']['vic_global_template'])
vic_forcing_basepath = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['VIC']['orig_forcing_nc_basepath'])
# --- run VIC with assigned states --- #
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
    



