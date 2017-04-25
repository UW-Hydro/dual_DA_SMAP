# =========================================================== #
# This script produces true and synthetic surface soil moisture measurements
#    - Run VIC with "truth" forcings and perturbed states --> "truth"
#    - Add random noise to "truth" top-layer soil moisture --> synthetic measurements
# =========================================================== #

import sys
import os
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import shutil

from tonic.models.vic.vic import VIC
from tonic.io import read_configobj

from da_utils import (Forcings, setup_output_dirs, propagate,
                      calculate_sm_noise_to_add_magnitude,
                      perturb_soil_moisture_states,
                      calculate_max_soil_moist_domain,
                      convert_max_moist_n_state, VarToPerturb,
                      find_global_param_value, propagate_linear_model,
                      concat_clean_up_history_file,
                      calculate_scale_n_whole_field,
                      calculate_cholesky_L)

# =========================================================== #
# Load command line arguments
# =========================================================== #
cfg = read_configobj(sys.argv[1])

# =========================================================== #
# Set random generation seed
# =========================================================== #
np.random.seed(cfg['CONTROL']['seed'])

# =========================================================== #
# Process some config parameters
# =========================================================== #
print('Processing config parameters...')
# Simulation time
start_time = pd.to_datetime(cfg['TIME_INDEX']['start_time'])
end_time = pd.to_datetime(cfg['TIME_INDEX']['end_time'])
start_year = start_time.year
end_year = end_time.year

# Set up output sub-directories
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_basedir']),
                         mkdirs=['truth', 'synthetic_meas', 'plots'])
truth_subdirs = setup_output_dirs(dirs['truth'],
                                  mkdirs=['global', 'history', 'states',
                                          'logs'])

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


# =========================================================== #
# Load specified truth history files
# =========================================================== #
print('Loading specified \"truth\" history files...')
list_ds = []
for year in range(start_year, end_year+1):
    ds = xr.open_dataset(os.path.join(
        cfg['CONTROL']['root_dir'],
        '{}{}.nc'.format(cfg['RESCALE_TRUTH']['truth_basepath'], year)))
    list_ds.append(ds)
ds_hist = xr.concat(list_ds, dim='time')
ds_hist.to_netcdf(os.path.join(
    truth_subdirs['history'],
    'history.concat.rescaled_v2.{}_{:05d}-{}_{:05d}.nc'.format(
        start_time.strftime('%Y%m%d'),
        start_time.hour*3600+start_time.second,
        end_time.strftime('%Y%m%d'),
        end_time.hour*3600+end_time.second)))


# =========================================================== #
# Simulate synthetic measurement - Extract top-layer soil
# moisture from "truth" at the end of each day, and add noise
# =========================================================== #

print('Simulating synthetic measurements...')

# Calculate maximum soil moisture for each tile [lat, lon, n]
global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])
da_max_moist = calculate_max_soil_moist_domain(global_template)

# --- Select out times of measurement --- #
ds_hist_meas_times = ds_hist.sel(time=meas_times)

# --- Select top-layer soil moisture --- #
da_sm1_true = ds_hist_meas_times['OUT_SOIL_MOIST'].sel(nlayer=0)

# --- Add noise --- #
# Generate the standard deviation of noise to be added for each grid cell
da_sigma = da_sm1_true[0, :, :].copy(deep=True)
da_sigma[:] = cfg['SYNTHETIC_MEAS']['sigma']
# Add noise
VarToPerturb_sm1 = VarToPerturb(da_sm1_true) # create class
da_sm1_perturbed = VarToPerturb_sm1.add_gaussian_white_noise(
                        da_sigma, da_max_moist.sel(nlayer=0),
                        adjust_negative=True)

# --- Save synthetic measurement to netCDF file --- #
ds_simulated = xr.Dataset({'simulated_surface_sm': da_sm1_perturbed})
ds_simulated.to_netcdf(os.path.join(dirs['synthetic_meas'],
                                    'synthetic_meas.rescaled_v2.{}_{}.nc'.format(
                                            start_time.strftime('%Y%m%d'),
                                            end_time.strftime('%Y%m%d'))),
                       format='NETCDF4_CLASSIC')

