

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys

from da_utils import rmse, innov_norm_var, setup_output_dirs
from tonic.io import read_config, read_configobj


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ======================================================== #
# Parameter setting
# ======================================================== #
# Time period
EnKF_start_time = pd.to_datetime(cfg['EnKF']['start_time'])
EnKF_end_time = pd.to_datetime(cfg['EnKF']['end_time'])

# For error maps
openloop_hist_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_EnKF_basedir'],
            'history',
            'history.openloop.{}-{:05d}.nc'.format(
            EnKF_start_time.strftime('%Y-%m-%d'),
            EnKF_start_time.hour*3600+EnKF_start_time.second))
postprocess_hist_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_postprocess_basedir'],
            'history',
            'history.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                EnKF_start_time.strftime('%Y%m%d'),
                EnKF_start_time.hour*3600+EnKF_start_time.second,
                EnKF_end_time.strftime('%Y%m%d'),
                EnKF_end_time.hour*3600+EnKF_end_time.second))
truth_hist_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['EnKF']['truth_hist_nc'])

# For innovation map
ens_hist_dir = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_EnKF_basedir'],
            'history',
            'EnKF_ensemble_concat')
meas_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['EnKF']['meas_nc'])
meas_sm1_varname = cfg['EnKF']['meas_var_name']
R = cfg['EnKF']['R']  # Measurement error variance
N = cfg['EnKF']['N']  # Number of ensemble members

# Output
output_basedir = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_postprocess_basedir'],
            'plots')

# ======================================================== #
# Setup output directory
# ======================================================== #
output_dir = setup_output_dirs(output_basedir,
                                mkdirs=['maps'])['maps']

# ======================================================== #
# Load and process data
# ======================================================== #
print('Loading data...')
# --- Load data --- #
ds_openloop = xr.open_dataset(openloop_hist_nc)
ds_postprocess = xr.open_dataset(postprocess_hist_nc)
ds_truth = xr.open_dataset(truth_hist_nc)
ds_meas = xr.open_dataset(meas_nc)
# Load ensemble data
list_ds_ens = []
for i in range(N):
    list_ds_ens.append(xr.open_dataset(os.path.join(
                ens_hist_dir,
                'history.ens{}.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                        i+1,
                        EnKF_start_time.strftime('%Y%m%d'),
                        EnKF_start_time.hour*3600+EnKF_start_time.second,
                        EnKF_end_time.strftime('%Y%m%d'),
                        EnKF_end_time.hour*3600+EnKF_end_time.second))))

# --- Cut data to the same time period --- #
start_time = EnKF_start_time
end_time = EnKF_end_time
ds_truth = ds_truth.sel(time=slice(start_time, end_time))
ds_openloop = ds_openloop.sel(time=slice(start_time, end_time))
ds_meas = ds_meas.sel(time=slice(start_time, end_time))
for i, ds in enumerate(list_ds_ens):
    ds = ds.sel(time=slice(start_time, end_time))
    list_ds_ens[i] = ds


# ======================================================== #
# Extract shared coordinates
# ======================================================== #
lat_coord = ds_openloop['lat']
lon_coord = ds_openloop['lon']
time_coord = ds_openloop['time']


# ======================================================== #
# Plot innovation - (y_meas - y_est_before_update)
# ======================================================== #
print('Calculating and plotting innovation...')
# --- Extract variables --- #
da_meas = ds_meas[meas_sm1_varname]
time_meas_coord = da_meas['time']
# Only keep timesteps of ens_mean where there are sm measurements
list_ens = []
for ds in list_ds_ens:
    print('aaa')
    array = ds['OUT_SOIL_MOIST'].sel(nlayer=0, time=time_meas_coord).values
    list_ens.append(array)

ens = np.array(list_ens)  # [N, time, lat, lon]
ens_mean = np.mean(ens, axis=0)  # [time, lat, lon]

# --- Calculate innovation (avg. over all ensemble members) time series
# for all grid cells --- #
innov = da_meas.values - ens_mean  # [time, lat, lon]
da_innov = xr.DataArray(innov, coords=[time_meas_coord, lat_coord, lon_coord],
                        dims=['time', 'lat', 'lon'])

# --- Calculate innovation statistics --- #
# (1) Mean
da_innov_mean = da_innov.mean(dim='time')
# (2) Normalized variance
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
innov = innov.reshape([len(time_meas_coord), nloop])  # [time, nloop]
y_est_before_update = ens.reshape(
                            [N, len(time_meas_coord), nloop])  # [N, time, nloop]
y_est_before_update = np.rollaxis(y_est_before_update, 0, 2)  # [time, N, nloop]
# Calculate normalized variance for all grid cells
var_norm = np.array(list(map(
            lambda j: innov_norm_var(innov[:, j], y_est_before_update[:, :, j], R),
            range(nloop))))  # [nloop]
# Reshape var_norm
var_norm = var_norm.reshape([len(lat_coord), len(lon_coord)])
# Put in da
da_var_norm = xr.DataArray(var_norm, coords=[lat_coord, lon_coord],
                           dims=['lat', 'lon'])

# --- Plot maps --- #
# Innovation mean
fig = plt.figure(figsize=(14, 7))
cs = da_innov_mean.plot(add_colorbar=False, cmap='bwr', vmin=-2, vmax=2)
cbar = plt.colorbar(cs, extend='both').set_label('Innovation (mm)', fontsize=20)
plt.title('Mean innovation (meas - y_est_before_update)\n'
          'Avg. value: {:.2f}'.format(float(da_innov_mean.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_dir, 'innov_mean.png'), format='png')

# Innovation normalized variance
fig = plt.figure(figsize=(14, 7))
cs = da_var_norm.plot(add_colorbar=False, cmap='bwr', vmin=0, vmax=2)
cbar = plt.colorbar(cs, extend='max').set_label('Normalized variance (mm2)', fontsize=20)
plt.title('Normalized innovation variance, '
          'avg. value: {:.2f}'.format(float(da_var_norm.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_dir, 'innov_var_norm.png'), format='png')


# ======================================================== #
# Plot error map - sm1
# ======================================================== #

# --- Extract variables --- #
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0)
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0)
da_postprocess = ds_postprocess['OUT_SOIL_MOIST'].sel(nlayer=0)

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(time_coord), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(time_coord), nloop])  # [time, nloop]
postprocess = da_postprocess.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_postprocess = np.array(list(map(
            lambda j: rmse(truth[:, j], postprocess[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_postprocess = rmse_postprocess.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_postprocess = xr.DataArray(rmse_postprocess, coords=[lat_coord, lon_coord],
                                   dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=10)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('sm1, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm1_openloop.png'), format='png')

# Postprocess
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_postprocess.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=10)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('sm1, RMSE of postprocess (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm1_postprocess.png'), format='png')

# Diff - (postprocess - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_postprocess - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-5, vmax=5)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE (mm)', fontsize=20)
plt.title('sm1, RMSE diff. (postprocess - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm1_diff.png'), format='png')


# ======================================================== #
# Plot error map - sm2
# ======================================================== #

# --- Extract variable --- #
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=1)
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=1)
da_postprocess = ds_postprocess['OUT_SOIL_MOIST'].sel(nlayer=1)

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(time_coord), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(time_coord), nloop])  # [time, nloop]
postprocess = da_postprocess.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_postprocess = np.array(list(map(
            lambda j: rmse(truth[:, j], postprocess[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_postprocess = rmse_postprocess.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_postprocess = xr.DataArray(rmse_postprocess, coords=[lat_coord, lon_coord],
                                   dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=300)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('sm2, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm2_openloop.png'), format='png')

# Postprocess
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_postprocess.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=300)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('sm2, RMSE of postprocess (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm2_postprocess.png'), format='png')

# Diff - (postprocess - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_postprocess - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-100, vmax=100)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE (mm)', fontsize=20)
plt.title('sm2, RMSE diff. (postprocess - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm2_diff.png'), format='png')


# ======================================================== #
# Plot error map - sm3
# ======================================================== #

# --- Extract variable --- #
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=2)
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=2)
da_postprocess = ds_postprocess['OUT_SOIL_MOIST'].sel(nlayer=2)

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(time_coord), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(time_coord), nloop])  # [time, nloop]
postprocess = da_postprocess.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_postprocess = np.array(list(map(
            lambda j: rmse(truth[:, j], postprocess[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_postprocess = rmse_postprocess.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_postprocess = xr.DataArray(rmse_postprocess, coords=[lat_coord, lon_coord],
                                   dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=300)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('sm3, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm3_openloop.png'), format='png')

# Postprocess
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_postprocess.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=300)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('sm3, RMSE of postprocess (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm3_postprocess.png'), format='png')

# Diff - (postprocess - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_postprocess - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-100, vmax=100)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE (mm)', fontsize=20)
plt.title('sm3, RMSE diff. (postprocess - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm3_diff.png'), format='png')


# ======================================================== #
# Plot error map - runoff
# ======================================================== #

# --- Extract variable --- #
da_truth = ds_truth['OUT_RUNOFF']
da_openloop = ds_openloop['OUT_RUNOFF']
da_postprocess = ds_postprocess['OUT_RUNOFF']

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(time_coord), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(time_coord), nloop])  # [time, nloop]
postprocess = da_postprocess.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_postprocess = np.array(list(map(
            lambda j: rmse(truth[:, j], postprocess[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_postprocess = rmse_postprocess.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_postprocess = xr.DataArray(rmse_postprocess, coords=[lat_coord, lon_coord],
                                   dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.3)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Runoff, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_runoff_openloop.png'), format='png')

# Postprocess
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_postprocess.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.3)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Runoff, RMSE of postprocess (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_runoff_postprocess.png'), format='png')

# Diff - (postprocess - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_postprocess - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-0.1, vmax=0.1)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Runoff, RMSE diff. (postprocess - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_runoff_diff.png'), format='png')


# ======================================================== #
# Plot error map - baseflow
# ======================================================== #

# --- Extract variable --- #
da_truth = ds_truth['OUT_BASEFLOW']
da_openloop = ds_openloop['OUT_BASEFLOW']
da_postprocess = ds_postprocess['OUT_BASEFLOW']

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(time_coord), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(time_coord), nloop])  # [time, nloop]
postprocess = da_postprocess.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_postprocess = np.array(list(map(
            lambda j: rmse(truth[:, j], postprocess[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_postprocess = rmse_postprocess.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_postprocess = xr.DataArray(rmse_postprocess, coords=[lat_coord, lon_coord],
                                   dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.3)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Baseflow, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_baseflow_openloop.png'), format='png')

# Postprocess
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_postprocess.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.3)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Baseflow, RMSE of postprocess (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_baseflow_postprocess.png'), format='png')

# Diff - (postprocess - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_postprocess - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-0.1, vmax=0.1)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Baseflow, RMSE diff. (postprocess - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_baseflow_diff.png'), format='png')


# ======================================================== #
# Plot error map - evap
# ======================================================== #

# --- Extract variable --- #
da_truth = ds_truth['OUT_EVAP']
da_openloop = ds_openloop['OUT_EVAP']
da_postprocess = ds_postprocess['OUT_EVAP']

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(time_coord), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(time_coord), nloop])  # [time, nloop]
postprocess = da_postprocess.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_postprocess = np.array(list(map(
            lambda j: rmse(truth[:, j], postprocess[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_postprocess = rmse_postprocess.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_postprocess = xr.DataArray(rmse_postprocess, coords=[lat_coord, lon_coord],
                                   dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.3)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Evap, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_evap_openloop.png'), format='png')

# Postprocess
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_postprocess.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.3)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Evap, RMSE of postprocess (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_evap_postprocess.png'), format='png')

# Diff - (postprocess - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_postprocess - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-0.1, vmax=0.1)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE (mm/step)', fontsize=20)
plt.title('Evap, RMSE diff. (postprocess - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_evap_diff.png'), format='png')

