
import matplotlib
matplotlib.use('Agg')
import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import sys


def rmse(true, est):
    ''' Calculates RMSE of an estimated variable compared to the truth variable

    Parameters
    ----------
    true: <np.array>
        A 1-D array of time series of true values
    est: <np.array>
        A 1-D array of time series of estimated values (must be the same length of true)

    Returns
    ----------
    rmse: <float>
        Root mean square error

    Require
    ----------
    numpy
    '''

    rmse = np.sqrt(sum((est - true)**2) / len(true))
    return rmse


def get_soil_depth(param_nc):
    '''Get soil depth for all grid cells and all soil layers (from soil parameters)

    Parameters
    ----------
    param_nc: <str>
        VIC input parameter netCDF file path (here it is only used to
        extract soil parameter file info)

    Returns
    ----------
    da_soil_depth: <xarray.DataArray>
        Soil depth for the whole domain and each soil layer [unit: mm];
        Dimension: [nlayer, lat, lon]

    Require
    ----------
    xarray
    '''

    # Load soil parameter file
    ds_soil = xr.open_dataset(param_nc, decode_cf=False)

    # Get soil depth for each layer
    # Dimension: [nlayer, lat, lon]
    da_soil_depth = ds_soil['depth']  # [m]
    # Convert unit to mm
    da_soil_depth = da_soil_depth * 1000 # [mm]

    return da_soil_depth


# ================================================= #
# Parameter setting
# ================================================= #
lat = 36.0625
lon = -102.0625

# Realizations of truth; "{}" will be replaced by random index 1, 2, 3, ...
truth_nc = '/civil/hydro/ymao/data_assim/output/synthetic_data/test.36.0625_-102.0625/' \
        'test.large_sm_pert.1980_1989/random{}/' \
        'truth/history/history.concat.19800101_00000-19891231_75600.nc'

# Openloop nc
openloop_nc = '/civil/hydro/ymao/data_assim/output/vic/ArkRed/openloop.1980_1989.Newman/history/history.openloop.1980-01-01-00000.nc'

# Forcings
start_year = 1980
end_year = 1989
# Realizations of truth forcings
# First "{}" will be replaced by random index 1, 2, 3, ...; second "{}" will be replaced by YYYY
truth_forcing_nc = '/civil/hydro/ymao/data_assim/output/ensemble_forcing/' \
            'test.36.0625_-102.0625/Newman_ens100_perturbed.prec_std_1.random{}/' \
            'ens_26/force.{}.nc'
# Openloop forcing
# "{}" will be replaced by YYYY
openloop_forcing_nc = '/civil/hydro/ymao/data_assim/forcing/vic/Newman/' \
                      'test.36.0625_-102.0625/ens_100/force.{}.nc'

# Ensemble members
N = 16

# VIC parameter netCDF file
vic_param_nc = '/civil/hydro/ymao/data_assim/param/vic/ArkRed/Naoki/ArkRed.param.nc'

# Output dir
output_dir = './output/20170210.test.large_sm_pert'

# ================================================= #
# Load data
# ================================================= #
# Load realizations of truth
print('Loading realizations of truth...')
list_ds_truth = []
for i in range(N):
    ds_truth = xr.open_dataset(truth_nc.format(i+1))
    list_ds_truth.append(ds_truth)

# Load openloop 
print('Loading openloop...')
ds_openloop = xr.open_dataset(openloop_nc)

# Load realizations of truth forcings
print('Loading realizations of truth forcings...')
list_da_truth_prec = []
for i in range(N):
    list_da = []
    for year in range(start_year, end_year):
        da = xr.open_dataset(truth_forcing_nc.format(i+1, year))['PREC']
        list_da.append(da)
    # Concat all years
    da_all_years = xr.concat(list_da, dim='time')
    list_da_truth_prec.append(da_all_years)
# Load openloop forcings
print('Loading openloop forcings...')
list_da = []
for year in range(start_year, end_year):
    da = xr.open_dataset(openloop_forcing_nc.format(year))
    list_da.append(da)
da_openloop_prec = xr.concat(list_da, dim='time')['PREC']

# ================================================= #
# Plot precip
# ================================================= #
print('Plotting precipitation...')
list_ts_truth = []
for i in range(N):
    da_truth = list_da_truth_prec[i]
    ts_truth = da_truth.sel(lat=lat, lon=lon).to_series()
    list_ts_truth.append(ts_truth)
ts_openloop = da_openloop_prec.sel(lat=lat, lon=lon).to_series()

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truth
for i in range(N):
    ts_truth = list_ts_truth[i]
    if i == 0:
        ts_truth.plot(color='grey', style='-', label='Truth', legend=True)
    else:
        ts_truth.plot(color='grey', style='-')

# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop',
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Precipitation (mm)')
plt.title('Precipitation, {}, {}'.format(lat, lon))

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.prec_truth_realizations.html'.format(lat, lon)))

p = figure(title='Precipitation, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Precipitation (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each realization
for i in range(N):
    if i == 0:
        legend="Truth"
    else:
        legend=False
    ts = list_ts_truth[i]
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop", line_width=2)
# Save
save(p)

# ================================================= #
# Process max soil moist
# ================================================= #
da_soil_depth = get_soil_depth(vic_param_nc).sel(lat=lat, lon=lon)  # [nlayers]
depth_sm1 = float(da_soil_depth[0].values)
depth_sm2 = float(da_soil_depth[1].values)
depth_sm3 = float(da_soil_depth[2].values)

# ================================================= #
# Plot sm1
# ================================================= #
print('Plotting sm1...')
list_ts_truth = []
for i in range(N):
    ds_truth = list_ds_truth[i]
    ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(lat=lat, lon=lon, nlayer=0).\
                to_series() / depth_sm1
    list_ts_truth.append(ts_truth)

ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(lat=lat, lon=lon, nlayer=0).\
                to_series() / depth_sm1
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm1_truth_realizations.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each realization
for i in range(N):
    if i == 0:
        legend="Truth"
    else:
        legend=False
    ts = list_ts_truth[i]
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop", line_width=2)
# Save
save(p)

# ================================================= #
# Plot sm2
# ================================================= #
print('Plotting sm2...')
list_ts_truth = []
for i in range(N):
    ds_truth = list_ds_truth[i]
    ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(lat=lat, lon=lon, nlayer=1).\
                to_series() / depth_sm2
    list_ts_truth.append(ts_truth)

ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(lat=lat, lon=lon, nlayer=1).\
                to_series() / depth_sm2
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truth
for i in range(N):
    ts_truth = list_ts_truth[i]
    if i == 0:
        ts_truth.plot(color='grey', style='-', label='Truth', legend=True, alpha=0.3)
    else:
        ts_truth.plot(color='grey', style='-', alpha=0.3)

# plot open-loop
ts_openloop.plot(color='m', style='--', lw=2,
                 label='Open-loop',
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moisture (mm/mm)')
plt.title('Middle-layer soil moisture, {}, {}'.format(lat, lon))
# Save figure
fig.savefig(os.path.join(output_dir,
                         '{}_{}.sm2_truth_realizations.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm2_truth_realizations.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each realization
for i in range(N):
    if i == 0:
        legend="Truth"
    else:
        legend=False
    ts = list_ts_truth[i]
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop", line_width=2)
# Save
save(p)

# ================================================= #
# Plot sm3
# ================================================= #
print('Plotting sm3...')
# sm3
list_ts_truth = []
for i in range(N):
    ds_truth = list_ds_truth[i]
    ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(lat=lat, lon=lon, nlayer=2).\
                to_series() / depth_sm3
    list_ts_truth.append(ts_truth)

ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(lat=lat, lon=lon, nlayer=2).\
                to_series() / depth_sm3
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truth
for i in range(N):
    ts_truth = list_ts_truth[i]
    if i == 0:
        ts_truth.plot(color='grey', style='-', label='Truth', legend=True, alpha=0.3)
    else:
        ts_truth.plot(color='grey', style='-', alpha=0.3)

# plot open-loop
ts_openloop.plot(color='m', style='--', lw=2,
                 label='Open-loop',
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moisture (mm/mm)')
plt.title('Bottom-layer soil moisture, {}, {}'.format(lat, lon))
# Save figure
fig.savefig(os.path.join(output_dir,
                         '{}_{}.sm3_truth_realizations.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm3_truth_realizations.html'.format(lat, lon)))

p = figure(title='Bottom-layer soil moisture, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each realization
for i in range(N):
    if i == 0:
        legend="Truth"
    else:
        legend=False
    ts = list_ts_truth[i]
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop", line_width=2)
# Save
save(p)



