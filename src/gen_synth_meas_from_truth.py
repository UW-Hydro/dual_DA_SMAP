
# This script add noise to synthetic true states to generate
# synthetic SM measurements

import os
import numpy as np
import pandas as pd
import xarray as xr
import numbers
import sys

from tonic.io import read_configobj

from da_utils import setup_output_dirs, VarToPerturb, calculate_max_soil_moist_domain


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


# --- Extract soil layer info --- #
# Calculate maximum soil moisture for each tile [lat, lon, n]
global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])
da_max_moist = calculate_max_soil_moist_domain(global_template)


# --- Load cellAvg "truth" states at measurement times --- #
print('\tLoading truth states...')
ds_state_cellAvg = xr.open_dataset(os.path.join(
    truth_subdirs['states'],
    'truth_state_cellAvg.{}_{}.nc'.format(
        meas_times[0].strftime('%Y%m%d'),
        meas_times[-1].strftime('%Y%m%d'))))


print('Generate synthetic measurements...')
# --- Select top-layer soil moisture --- #
da_sm1_true = ds_state_cellAvg['SOIL_MOISTURE'].sel(nlayer=0)

# --- Add noise --- #
# If input is a numerical number, assign spatial constant R values
if isinstance(cfg['SYNTHETIC_MEAS']['R'], numbers.Number):
    sigma = np.empty([len(da_sm1_true['lat']), len(da_sm1_true['lon'])])
    sigma[:] = np.sqrt(cfg['SYNTHETIC_MEAS']['R'])
# If input is an xr.Dataset
else:
    if cfg['SYNTHETIC_MEAS']['R_vartype'] == 'R':
        sigma = np.sqrt(xr.open_dataset(
                    os.path.join(cfg['CONTROL']['root_dir'], cfg['SYNTHETIC_MEAS']['R']))\
                    [cfg['SYNTHETIC_MEAS']['R_varname']].values)
    elif cfg['SYNTHETIC_MEAS']['R_vartype'] == 'std':
        sigma = xr.open_dataset(
            os.path.join(cfg['CONTROL']['root_dir'], cfg['SYNTHETIC_MEAS']['R']))\
            [cfg['SYNTHETIC_MEAS']['R_varname']].values
# Put into a da
da_sigma = da_sm1_true[0, :, :].copy(deep=True)
da_sigma[:] = sigma
# Generate the standard deviation of noise to be added for each grid cell
# Add noise
VarToPerturb_sm1 = VarToPerturb(da_sm1_true) # create class
da_sm1_perturbed = VarToPerturb_sm1.add_gaussian_white_noise(
                        da_sigma, da_max_moist.sel(nlayer=0),
                        phi=cfg['SYNTHETIC_MEAS']['phi'],
                        adjust_negative=True)


# --- Save synthetic measurement to netCDF file --- #
ds_simulated = xr.Dataset({'simulated_surface_sm': da_sm1_perturbed})
ds_simulated.to_netcdf(
    os.path.join(
        dirs['synthetic_meas'],
        '{}{}_{}.nc'.format(cfg['OUTPUT']['meas_nc_prefix'],
                             start_time.strftime('%Y%m%d'),
                            end_time.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')



