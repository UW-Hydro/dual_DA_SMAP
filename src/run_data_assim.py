
''' This script performs EnKF assimilation of soil moisture into VIC states,
    based on original precipitation forcing.

    Usage:
        $ python run_data_assim.py config_file nproc mpi_proc debug
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
from da_utils import (EnKF_VIC, setup_output_dirs, generate_VIC_global_file,
                      check_returncode, propagate, calculate_ensemble_mean_states,
                      run_vic_assigned_states, concat_vic_history_files,
                      calculate_sm_noise_to_add_magnitude,
                      calculate_sm_noise_to_add_covariance_matrix_whole_field,
                      calculate_max_soil_moist_domain,
                      convert_max_moist_n_state)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Number of processors for parallelizing ensemble runs
nproc = int(sys.argv[2])

# Number of processors for each VIC run
mpi_proc = int(sys.argv[3])

# Whether to print out debug temp files or not
debug = (sys.argv[4].lower() == 'true')

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
                                 'logs', 'plots', 'temp'])


# ============================================================ #
# Prepare VIC exe and MPI exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))
mpi_exe = cfg['VIC']['mpi_exe']


# ============================================================ #
# Prepare and run EnKF
# ============================================================ #
print('Preparing for running EnKF...')

# --- Load and process measurement data --- #
print('\tLoading measurement data...')
# Load measurement data
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

# --- Prepare measurement error covariance matrix R [m*m] --- #
R = np.array([[cfg['EnKF']['R']]])

# --- Calculate state perturbation covariance matrix --- #
# Calculate perturvation magnitude [nlayer, lat, lon]
da_scale = calculate_sm_noise_to_add_magnitude(
                vic_history_path=os.path.join(
                        cfg['CONTROL']['root_dir'],
                        cfg['EnKF']['vic_history_path']),
                sigma_percent=cfg['EnKF']['state_perturb_sigma_percent'])
# Extract veg_class and snow_band information from a state file
init_state_nc = os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['EnKF']['vic_initial_state'])
ds_state = xr.open_dataset(init_state_nc)
nveg = len(ds_state['veg_class'])
nsnow= len(ds_state['snow_band'])
# Calculate covariance matrix
P_whole_field = calculate_sm_noise_to_add_covariance_matrix_whole_field(
                    da_scale, nveg, nsnow,
                    cfg['EnKF']['state_perturb_corrcoef'])
# Calculate maximum soil moisture for each tile [lat, lon, n]
da_max_moist = calculate_max_soil_moist_domain(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['VIC']['vic_global_template']))
da_max_moist_n = convert_max_moist_n_state(da_max_moist, nveg, nsnow)

# --- Run EnKF --- #
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])
print('Start running EnKF for ', start_time, 'to', end_time, '...')

dict_ens_list_history_files = EnKF_VIC(
         N=cfg['EnKF']['N'],
         start_time=start_time,
         end_time=end_time,
         init_state_nc=os.path.join(cfg['CONTROL']['root_dir'],
                                    cfg['EnKF']['vic_initial_state']),
         P_whole_field=P_whole_field,
         da_max_moist_n=da_max_moist_n,
         R=R,
         da_meas=da_meas,
         da_meas_time_var='time',
         vic_exe=vic_exe,
         vic_global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['VIC']['vic_global_template']),
         ens_forcing_basedir=os.path.join(cfg['CONTROL']['root_dir'],
                                           cfg['FORCINGS']['ens_forcing_basedir']),
         ens_forcing_prefix=cfg['FORCINGS']['ens_forcing_prefix'],
         vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
         output_vic_global_root_dir=dirs['global'],
         output_vic_state_root_dir=dirs['states'],
         output_vic_history_root_dir=dirs['history'],
         output_vic_log_root_dir=dirs['logs'],
         dict_varnames=dict_varnames,
         nproc=nproc,
         debug=debug,
         output_temp_dir=dirs['temp'])

# --- Concatenate all history files for each ensemble and clean up --- #
out_hist_concat_dir = setup_output_dirs(dirs['history'],
                                        mkdirs=['EnKF_ensemble_concat'])\
                      ['EnKF_ensemble_concat']

dict_ens_hist_concat = {}  # a dictionary of concatenated hist file for each ensemble member
for i in range(cfg['EnKF']['N']):
    # Concatenate
    ds_concat = concat_vic_history_files(dict_ens_list_history_files['ens{}'.format(i+1)])
    hist_concat_path = os.path.join(
                            out_hist_concat_dir,
                            'history.ens{}.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                i+1,
                                start_time.strftime('%Y%m%d'),
                                start_time.hour*3600+start_time.second,
                                end_time.strftime('%Y%m%d'),
                                end_time.hour*3600+end_time.second))
    ds_concat.to_netcdf(hist_concat_path, format='NETCDF4_CLASSIC')
    dict_ens_hist_concat['ens{}'.format(i+1)] = hist_concat_path
    # Clean up individual history files
    for f in dict_ens_list_history_files['ens{}'.format(i+1)]:
        os.remove(f)


