
''' This script post processes EnKF updated soil moisture states and SMART
    corrected rainfall. Specifically:
        1) Take average of all ensemble updated states
        2) Insert the updated states, and use corrected rainfall forcing to run VIC

    Usage:
        $ python postprocess_EnKF.py <config_file> <nproc> <mpi_proc>
'''

from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import xarray as xr
import sys
import multiprocessing as mp
import shutil

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic import VIC
from da_utils import (setup_output_dirs, run_vic_assigned_states,
                      Forcings, to_netcdf_forcing_file_compress)

# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Read number of processors for VIC MPI runs
mpi_proc = int(sys.argv[2])

# Ensemble index of prec to postprocess (index starts from 1)
# Options:
#   integer index of SMART-corrected prec (prec file name should be: "prec_corrected.ens<i>.YYYY.nc");
#   "mean" for SMART-corrected ensemble-mean (prec file name: "prec_corrected.YYYY.nc")
#   "orig" for using the original prec forcing for postprocessing
#   "true" for using the true forcing for postprocessing
ens_prec = sys.argv[3]

# Ensemble index of updated states (index starts from 1)
ens_state = sys.argv[4]


# ============================================================ #
# Prepare VIC exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))


# ============================================================ #
# Process cfg data
# ============================================================ #
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])

start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Setup postprocess output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(
                            cfg['CONTROL']['root_dir'],
                            cfg['POSTPROCESS']['output_postprocess_basedir']),
                         mkdirs=['global', 'history', 'forcings',
                                 'logs', 'plots'])


# ============================================================ #
# Load data
# ============================================================ #
# --- Load measurement data --- #
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']] # Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(start_time, end_time))
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])


# ============================================================ #
# Post-process fluxes - run VIC from EnKF_start_time to
# EnKF_end_time.
# States:
#   EnKF updated states; either keep the ensemble of updated
#   states, or only use the ensemble mean updated states
#   (specified in the cfg file).
# Forcings:
#   Specified prec forcing (other forcing variables are from orig. forcings)
# ============================================================ #
# ----------------------------------------------------------------- #
# --- Generate forcings for post-processing - replace prec data --- #
# --- in the original forcing file                              --- #
# ----------------------------------------------------------------- #
print('Replacing precip data...')
# If use original forcing for post-processing
if ens_prec == 'orig':
    vic_forcing_basepath = os.path.join(
        cfg['CONTROL']['root_dir'], cfg['FORCINGS']['orig_forcing_nc_basepath'])
# If use the true forcing for post-processing
elif ens_prec == 'true':
    vic_forcing_basepath = os.path.join(
        cfg['CONTROL']['root_dir'], cfg['FORCINGS']['truth_forcing_nc_basepath'])
# If use SMART-corrected prec
else:
    for year in range(start_year, end_year+1):
        # Load prec data
        if ens_prec == 'mean':
            da_prec = xr.open_dataset(
                os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['POSTPROCESS']['SMART_outdir'],
                             'prec_corrected.{}.nc'.format(year)))\
                ['prec_corrected']
        else:
            da_prec = xr.open_dataset(
                os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['POSTPROCESS']['SMART_outdir'],
                             'prec_corrected.ens{}.{}.nc'.format(ens_prec, year)))\
                ['prec_corrected']
        # Load in orig forcings
        class_forcings_orig = Forcings(xr.open_dataset('{}{}.nc'.format(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['FORCINGS']['orig_forcing_nc_basepath']),
                    year)))
        # Replace prec
        ds_prec_replaced = class_forcings_orig.replace_prec(
                                cfg['FORCINGS']['PREC'],
                                da_prec)
        # Save replaced forcings to netCDF file
        if ens_prec == 'mean':
            to_netcdf_forcing_file_compress(
                ds_prec_replaced,
                out_nc=os.path.join(
                    dirs['forcings'],
                    'forc.post_prec.ens_mean.{}.nc'.format(year)))
        else:
            to_netcdf_forcing_file_compress(
                ds_prec_replaced,
                out_nc=os.path.join(
                    dirs['forcings'],
                    'forc.post_prec.ens{}.{}.nc'.format(ens_prec, year)))
    # Set VIC forcing
    vic_forcing_basepath = os.path.join(
                dirs['forcings'],
                'forc.post_prec.ens{}.'.format(ens_prec))


# ----------------------------------------------------------------- #
# --- Run VIC with assinged states --- #
# ----------------------------------------------------------------- #
print('Post-process - run VIC with updated states...')
# --- Prepare some variables --- #
# initial state nc
init_state_nc = os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['OUTPUT']['output_EnKF_basedir'],
    'states',
    'init.{}_{:05d}'.format(
            start_time.strftime('%Y%m%d'),
            start_time.hour*3600+start_time.second),
    'state.ens{}.nc'.format(ens_state))
# state file dict
dict_assigned_state_nc = OrderedDict()
for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
    state_time = pd.to_datetime(time)
    dict_assigned_state_nc[state_time] = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_EnKF_basedir'],
        'states',
        'updated.{}_{:05d}'.format(
                time.strftime('%Y%m%d'),
                time.hour*3600+time.second),
        'state.ens{}.nc'.format(ens_state))
# make subdirs for global, history and log files for
# each ensemble member
if ens_prec == 'orig':
    subdir_name = 'force_orig.state_ens{}'.format(ens_state)
elif ens_prec == 'true':
    subdir_name = 'force_truth.state_ens{}'.format(ens_state)
elif ens_prec == 'mean':
    subdir_name = 'force_mean.state_ens{}'.format(ens_state)
else:
    subdir_name = 'force_ens{}.state_ens{}'.format(ens_prec, ens_state)
hist_subdir = setup_output_dirs(
                    dirs['history'],
                    mkdirs=[subdir_name])\
              [subdir_name]
global_subdir = setup_output_dirs(
                    dirs['global'],
                    mkdirs=[subdir_name])\
              [subdir_name]
log_subdir = setup_output_dirs(
                    dirs['logs'],
                    mkdirs=[subdir_name])\
              [subdir_name]
# other variables
global_template = os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['VIC']['vic_global_template'])
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
    output_global_root_dir=global_subdir,
    output_vic_history_root_dir=hist_subdir,
    output_vic_log_root_dir=log_subdir,
    mpi_proc=mpi_proc,
    mpi_exe=cfg['VIC']['mpi_exe'])
# --- Concat all years and clean up --- #
list_ds_hist = [xr.open_dataset(f) for f in list_history_files]
ds_concat = xr.concat(list_ds_hist, dim='time')
to_netcdf_forcing_file_compress(
    ds_concat,
    out_nc=os.path.join(
        hist_subdir,
        'history.concat.{}_{}.nc'.format(start_year, end_year)))
for f in list_history_files:
    os.remove(f)


