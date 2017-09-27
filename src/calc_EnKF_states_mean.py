
''' This script calculates average of EnKF updated states across all ensemble members.

    Usage:
        $ python calc_EnKF_states_mean.py <config_file (of run_data_assim)>
'''

import sys
import os
import xarray as xr
import pandas as pd
from collections import OrderedDict

from tonic.io import read_config, read_configobj
from da_utils import calculate_ensemble_mean_states

# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Process cfg data
# ============================================================ #
N = cfg['EnKF']['N']  # number of ensemble members
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])

start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Load measurement data
# ============================================================ #
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
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
# Identify EnKF result directories
# ============================================================ #
# --- Identify EnKF result directories --- #
EnKF_dirs = OrderedDict()
EnKF_basedir = os.path.join(cfg['CONTROL']['root_dir'],
                            cfg['OUTPUT']['output_EnKF_basedir'])
EnKF_dirs['states'] = os.path.join(EnKF_basedir, 'states')
EnKF_dirs['history'] = os.path.join(EnKF_basedir, 'history')
EnKF_dirs['global'] = os.path.join(EnKF_basedir, 'global')
EnKF_dirs['logs'] = os.path.join(EnKF_basedir, 'logs')


# ============================================================ #
# Calculate the ensemble-mean of the updated states
# ============================================================ #
print('Calculating ensemble-mean of the updates states...')

# --- Calculate ensemble-mean for the initial time point --- #
init_time = start_time
# Create a list of state file nc paths
state_dir = os.path.join(EnKF_dirs['states'], 'init.{}_{:05d}'.format(
                                init_time.strftime('%Y%m%d'),
                                init_time.hour*3600+init_time.second))
list_state_nc = []
for i in range(N):
    list_state_nc.append(os.path.join(state_dir, 'state.ens{}.nc'.format(i+1)))
# Calculate ensemble-mean states
print('\tInitial time')
init_state_mean_nc = calculate_ensemble_mean_states(
                            list_state_nc,
                            out_state_nc=os.path.join(state_dir, 'state.ens_mean.nc'))

# Loop over each measurement time point of updates states
dict_assigned_state_nc = OrderedDict()  #  An ordered dict of state times and nc files after the initial time
for t, time in enumerate(pd.to_datetime(da_meas['time'].values)):
    state_time = pd.to_datetime(time)
    print('\t', state_time)
    # Create a list of state file nc paths
    state_dir = os.path.join(EnKF_dirs['states'], 'updated.{}_{:05d}'.format(
                                state_time.strftime('%Y%m%d'),
                                state_time.hour*3600+state_time.second))
    list_state_nc = []
    for i in range(N):
        list_state_nc.append(os.path.join(state_dir,
                                          'state.ens{}.nc'.format(i+1)))
    # Calculate ensemble-mean states
    dict_assigned_state_nc[time] = calculate_ensemble_mean_states(
        list_state_nc,
        out_state_nc=os.path.join(state_dir, 'state.ens_mean.nc'))

