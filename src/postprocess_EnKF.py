
''' This script post processes EnKF updated soil moisture states and SMART
    corrected rainfall. Specifically:
        1) Take average of all ensemble updated states
        2) Insert the updated states, and use corrected rainfall forcing to run VIC

    Usage:
        $ python postprocess_EnKF.py <config_file> <mpi_proc>
'''

from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import xarray as xr
import sys

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
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))


# ============================================================ #
# Load data
# ============================================================ #
# --- Load measurement data --- #
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
# Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(cfg['EnKF']['start_time'], cfg['EnKF']['end_time']))
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])


# ============================================================ #
# Calculate the ensemble-mean of the updated states
# ============================================================ #
print('Calculating ensemble-mean of the updates states...')

# --- Determine the output directories for ensemble-mean states --- #
dirs = OrderedDict()
out_basedir = os.path.join(cfg['CONTROL']['root_dir'],
                           cfg['OUTPUT']['output_EnKF_basedir'])
dirs['states'] = os.path.join(out_basedir, 'states')
dirs['history'] = os.path.join(out_basedir, 'history')
dirs['global'] = os.path.join(out_basedir, 'global')
dirs['logs'] = os.path.join(out_basedir, 'logs')


N = cfg['EnKF']['N']  # number of ensemble members

# --- Calculate ensemble-mean for the initial time point --- #
init_time = pd.to_datetime(cfg['EnKF']['start_time'])
# Create a list of state file nc paths
state_dir = os.path.join(dirs['states'], 'init.{}_{:05d}'.format(
                                init_time.strftime('%Y%m%d'),
                                init_time.hour*3600+init_time.second))
list_state_nc = []
for i in range(N):
    list_state_nc.append(os.path.join(state_dir, 'state.ens{}.nc'.format(i+1)))
# Calculate ensemble-mean states
init_state_mean_nc = calculate_ensemble_mean_states(
                            list_state_nc,
                            out_state_nc=os.path.join(state_dir, 'state.ens_mean.nc'))

# Loop over each measurement time point of updates states
dict_assigned_state_nc = OrderedDict()  #  An ordered dict of state times and nc files after the initial time
for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
    state_time = pd.to_datetime(time)
    print('\t', state_time)
    # Create a list of state file nc paths
    state_dir = os.path.join(dirs['states'], 'updated.{}_{:05d}'.format(
                                state_time.strftime('%Y%m%d'),
                                state_time.hour*3600+state_time.second))
    list_state_nc = []
    for i in range(N):
        list_state_nc.append(os.path.join(state_dir, 'state.ens{}.nc'.format(i+1)))
    # Calculate ensemble-mean states
    dict_assigned_state_nc[time] = calculate_ensemble_mean_states(
                                        list_state_nc,
                                        out_state_nc=os.path.join(state_dir, 'state.ens_mean.nc'))

# ============================================================ #
# Post-process fluxes - run VIC with ensemble-mean updated
# states from EnKF start_time to EnKF end_time, and with
# specified prec forcing (other forcing variables are from orig. forcings)
# ============================================================ #
print('Post-process - run VIC with ensemble-mean updated states...')

# Set up output directories for post-processing results
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_postprocess_basedir']),
                         mkdirs=['global', 'history', 'states', 'forcings',
                                 'logs', 'plots'])

# Generate forcings for post-processing - replace prec data
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])
start_year = start_time.year
end_year = end_time.year

for year in range(start_year, end_year+1):
    # Load prec data
    da_prec = xr.open_dataset('{}{}.nc'.format(
                os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['FORCINGS']['prec_postprocess_nc_basepath']),
                year))[cfg['FORCINGS']['prec_postprocess_varname']]
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
    ds_prec_replaced.to_netcdf(os.path.join(
                        dirs['forcings'],
                        'forc.post_prec.{}.nc'.format(year)),
                     format='NETCDF4_CLASSIC')

# Run VIC with assinged states
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])

list_history_files = run_vic_assigned_states(
                        start_time=start_time, end_time=end_time,
                        vic_exe=vic_exe, init_state_nc=init_state_mean_nc,
                        dict_assigned_state_nc=dict_assigned_state_nc,
                        global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                                     cfg['VIC']['vic_global_template']),
                        vic_forcing_basepath=os.path.join(
                                    dirs['forcings'],
                                    'forc.post_prec.'),
                        vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
                        output_global_root_dir=dirs['global'],
                        output_state_root_dir=dirs['states'],
                        output_vic_history_root_dir=dirs['history'],
                        output_vic_log_root_dir=dirs['logs'],
                        mpi_proc=mpi_proc,
                        mpi_exe=cfg['VIC']['mpi_exe'])

# Concatenate all history files
ds_concat = concat_vic_history_files(list_history_files)
hist_ens_mean_post = os.path.join(
                        dirs['history'],
                        'history.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                start_time.strftime('%Y%m%d'),
                                start_time.hour*3600+start_time.second,
                                end_time.strftime('%Y%m%d'),
                                end_time.hour*3600+end_time.second))
ds_concat.to_netcdf(hist_ens_mean_post, format='NETCDF4_CLASSIC')

# Clean up individual histroy files
for f in list_history_files:
    os.remove(f)


