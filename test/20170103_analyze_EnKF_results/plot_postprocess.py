
# Usage:
#   python plot_postprocess.py <config_file>

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

from tonic.io import read_config, read_configobj


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


def find_global_param_value(gp, param_name, second_param=False):
    ''' Return the value of a global parameter

    Parameters
    ----------
    gp: <str>
        Global parameter file, read in by read()
    param_name: <str>
        The name of the global parameter to find
    second_param: <bool>
        Whether to read a second value for the parameter (e.g., set second_param=True to
        get the snowband param file path when SNOW_BAND>1)

    Returns
    ----------
    line_list[1]: <str>
        The value of the global parameter
    (optional) line_list[2]: <str>
        The value of the second value in the global parameter file when second_param=True
    '''
    for line in iter(gp.splitlines()):
        line_list = line.split()
        if line_list == []:
            continue
        key = line_list[0]
        if key == param_name:
            if second_param == False:
                return line_list[1]
            else:
                return line_list[1], line_list[2]


def determine_tile_frac(global_path):
    ''' Determines the fraction of each veg/snowband tile in each grid cell based on VIC
        global and parameter files

    Parameters
    ----------
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract snowband and vegparam files/options)

    Returns
    ----------
    da_tile_frac: <xr.DataArray>
        Fraction of each veg/snowband in each grid cell for the whole domain
        Dimension: [veg_class, snow_band, lat, lon]

    Require
    ----------
    numpy
    xarray
    '''

    # --- Load global parameter file --- #
    with open(global_path, 'r') as global_file:
            global_param = global_file.read()

    # --- Extract Cv from vegparam file (as defined in the global file) --- #
    param_nc = find_global_param_value(global_param, 'PARAMETERS')
    ds_param = xr.open_dataset(param_nc, decode_cf=False)
    da_Cv = ds_param['Cv']  # dim: [veg_class, lat, lon]
    lat = da_Cv['lat']
    lon = da_Cv['lon']

    # --- Extract snowband info from the global and param files --- #
    SNOW_BAND = find_global_param_value(global_param, 'SNOW_BAND')
    if SNOW_BAND.upper() == 'TRUE':
        n_snowband = len(ds_param['snow_band'])
    else:
        n_snowband = 1
    # Dimension of da_AreaFract: [snowband, lat, lon]
    if n_snowband == 1:  # if only one snowband
        data = np.ones([1, len(lat), len(lon)])
        da_AreaFract = xr.DataArray(data, coords=[[0], lat, lon],
                                    dims=['snow_band', 'lat', 'lon'])
    else:  # if more than one snowband
        da_AreaFract = ds_param['AreaFract']

    # --- Initialize the final DataArray --- #
    veg_class = da_Cv['veg_class']
    snow_band = da_AreaFract['snow_band']
    data = np.empty([len(veg_class), len(snow_band), len(lat), len(lon)])
    data[:] = np.nan
    da_tile_frac = xr.DataArray(data, coords=[veg_class, snow_band, lat, lon],
                                dims=['veg_class', 'snow_band', 'lat', 'lon'])

    # --- Calculate fraction of each veg/snowband tile for each grid cell,
    # and fill in da_file_frac --- #
    # Determine the total number of loops
    nloop = len(lat) * len(lon)
    # Convert Cv and AreaFract to np.array and straighten lat and lon into nloop
    Cv = da_Cv.values.reshape([len(veg_class), nloop])  # [nveg, nloop]
    AreaFract = da_AreaFract.values.reshape([len(snow_band), nloop])  # [nsnow, nloop]

    # Multiply Cv and AreaFract for each tile and grid cell
    tile_frac = np.array(list(map(
                    lambda i: np.dot(
                        Cv[:, i].reshape([len(veg_class), 1]),
                        AreaFract[:, i].reshape([1, len(snow_band)])),
                    range(nloop))))  # [nloop, nveg, nsnow]

    # Reshape tile_frac
    tile_frac = np.rollaxis(tile_frac, 0, 3)  # [nveg, nsow, nloop]
    tile_frac = tile_frac.reshape([len(veg_class), len(snow_band), len(lat), len(lon)])

    # Put in da_tile_frac
    da_tile_frac[:] = tile_frac

    return da_tile_frac


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


# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.argv[1])

# ========================================================== #
# Parameter setting
# ========================================================== #

if 'LINEAR_MODEL' in cfg:
    linear_model = True
else:
    linear_model = False

lat = cfg['GRID_CELL']['lat']
lon = cfg['GRID_CELL']['lon']

# --- Input directory and files --- #
# Post-process results
post_result_basedir = cfg['POSTPROCESS']['post_result_basedir']

# gen_synthetic results
gen_synth_basedir = cfg['EnKF']['gen_synth_basedir']
truth_nc_filename = cfg['EnKF']['truth_nc_filename']
synth_meas_nc_filename = cfg['EnKF']['synth_meas_nc_filename']

# openloop
openloop_nc = cfg['EnKF']['openloop_nc']

# initial time
init_time = pd.to_datetime(cfg['EnKF']['init_time'])

# VIC global file template (for extracting param file and snow_band)
vic_global_txt = cfg['EnKF']['vic_global_txt']

# Forcings (for all basepaths, 'YYYY.nc' will be appended)
orig_force_basepath = cfg['EnKF']['orig_force_basepath']
truth_force_basepath = cfg['EnKF']['truth_force_basepath']
# ens_force_basedir/ens_<i>/force.<YYYY>.nc, where <i> = 1, 2, ..., N
ens_force_basedir = cfg['EnKF']['ens_force_basedir']

# VIC parameter netCDF file
vic_param_nc = cfg['EnKF']['vic_param_nc']
    
# --- Measurement times --- #
times = pd.date_range(cfg['EnKF']['meas_start_time'],
                      cfg['EnKF']['meas_end_time'],
                      freq=cfg['EnKF']['freq'])
ntime = len(times)

# --- Plot time period --- #
plot_start_time = pd.to_datetime(cfg['EnKF']['plot_start_time'])
plot_end_time = pd.to_datetime(cfg['EnKF']['plot_end_time'])
start_year = plot_start_time.year
end_year = plot_end_time.year

# --- others --- #
N = cfg['EnKF']['N']  # number of ensemble members
ens = cfg['EnKF']['ens']  # index of ensemble member to plot for debugging plots

# --- Output --- #
output_dir = cfg['POSTPROCESS']['output_post_dir']


# ========================================================== #
# Load data
# ========================================================== #
print('Loading data...')

# --- Openloop --- #
print('\tOpenloop...')
ds_openloop = xr.open_dataset(openloop_nc)

# --- Truth --- #
print('\tTruth...')
ds_truth = xr.open_dataset(os.path.join(gen_synth_basedir, 'truth', 
                                        'history', truth_nc_filename))

# --- Measurements --- #
print('\tMeasurements...')
ds_meas = xr.open_dataset(os.path.join(gen_synth_basedir, 'synthetic_meas',
                                       synth_meas_nc_filename))

# --- Postprocessing results --- #
print('\tPostprocessing results...')
list_ds_post = []
for i in range(N):
    list_ds = []
    for year in range(start_year, end_year+1):
        # Load data
        fname = os.path.join(
            post_result_basedir,
            'history',
            'ens{}'.format(i+1),
            'history.concat.{}.nc'.format(year))
        ds = xr.open_dataset(fname).sel(lat=lat, lon=lon)
        list_ds.append(ds)
    # Concat all years
    ds_all_years = xr.concat(list_ds, dim='time')
    # Put data in list
    list_ds_post.append(ds_all_years)
# Concat all ensemble members together
ds_post = xr.concat(list_ds_post, dim='N')
ds_post['N'] = range(1, N+1)


# ========================================================== #
# Plot - sm1
# ========================================================== #
print('\tPlot - sm1...')
da_soil_depth = get_soil_depth(vic_param_nc).sel(lat=lat, lon=lon)  # [nlayers]
depth_sm1 = float(da_soil_depth[0].values)

# --- RMSE --- #
# extract time series
ts_meas = ds_meas['simulated_surface_sm'].sel(
                lat=lat, lon=lon, time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=0,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
da_post = ds_post['OUT_SOIL_MOIST'].sel(nlayer=0,
                                        time=slice(plot_start_time, plot_end_time)) / depth_sm1
ts_post_mean = da_post.mean(dim='N').\
               to_series()
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=0,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
rmse_meas = rmse(df_truth_meas['truth'].values, df_truth_meas['meas'])
# Calculate post_mean vs. truth
df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot each ensemble member
for i in range(N):
    if i == 0:
        legend=True
    else:
        legend=False
    da_post.sel(N=i+1).to_series().plot(
                color='blue', style='-', alpha=0.3,
                label='Post-processed, mean RMSE={:.3f} mm/mm'.format(rmse_post_mean),
                legend=legend)
# plot measurement
ts_meas.plot(style='ro', label='Measurement, RMSE={:.3f} mm/mm'.format(rmse_meas),
             legend=True)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.3f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Top-layer soil moisture, {}, {}, N={}'.format(lat, lon, N))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.post.sm1.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.post.sm1.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each ensemble member
for i in range(N):
    ens_name = 'ens{}'.format(i+1)
    if i == 0:
        legend="Post-processed, mean RMSE={:.2f} mm/mm".format(rmse_post_mean)
    else:
        legend=False
    ts = da_post.sel(N=i+1).to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot measurement
ts = ts_meas
p.circle(ts.index, ts.values, color="red", fill_color="red",
         legend="Measurement, RMSE={:.3f} mm/mm".format(rmse_meas), line_width=2)
# plot truth
ts = ts_truth
p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop, RMSE={:.3f} mm/mm".format(rmse_openloop), line_width=2)
# Save
save(p)

# ========================================================== #
# Plot - sm2
# ========================================================== #
print('\tPlot - sm2...')
depth_sm2 = float(da_soil_depth[1].values)

# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=1,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm2
da_post = ds_post['OUT_SOIL_MOIST'].sel(nlayer=1,
                                        time=slice(plot_start_time, plot_end_time)) / depth_sm2
ts_post_mean = da_post.mean(dim='N').\
               to_series()
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=1,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm2
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
# Calculate post_mean vs. truth
df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot each ensemble member
for i in range(N):
    if i == 0:
        legend=True
    else:
        legend=False
    da_post.sel(N=i+1).to_series().plot(
                color='blue', style='-', alpha=0.3,
                label='Post-processed, mean RMSE={:.5f} mm/mm'.format(rmse_post_mean),
                legend=legend)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.5f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Middle-layer soil moisture, {}, {}, N={}'.format(lat, lon, N))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.post.sm2.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.post.sm2.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each ensemble member
for i in range(N):
    ens_name = 'ens{}'.format(i+1)
    if i == 0:
        legend="Post-processed, mean RMSE={:.5f} mm/mm".format(rmse_post_mean)
    else:
        legend=False
    ts = da_post.sel(N=i+1).to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot truth
ts = ts_truth
p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop, RMSE={:.5f} mm/mm".format(rmse_openloop), line_width=2)
# Save
save(p)

# ========================================================== #
# Plot - sm3
# ========================================================== #
print('\tPlot - sm3...')
depth_sm3 = float(da_soil_depth[2].values)

# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=2,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm3
da_post = ds_post['OUT_SOIL_MOIST'].sel(nlayer=2,
                                        time=slice(plot_start_time, plot_end_time)) / depth_sm3
ts_post_mean = da_post.mean(dim='N').\
               to_series()
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=2,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm3
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
# Calculate post_mean vs. truth
df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot each ensemble member
for i in range(N):
    if i == 0:
        legend=True
    else:
        legend=False
    da_post.sel(N=i+1).to_series().plot(
                color='blue', style='-', alpha=0.3,
                label='Post-processed, mean RMSE={:.5f} mm/mm'.format(rmse_post_mean),
                legend=legend)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.5f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Bottom-layer soil moisture, {}, {}, N={}'.format(lat, lon, N))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.post.sm3.png'.format(lat, lon)),
            format='png')

# ========================================================== #
# Plot - runoff
# ========================================================== #
if not linear_model:
    print('\tPlot - surface runoff...')
    # --- RMSE --- #
    # extract time series
    ts_truth = ds_truth['OUT_RUNOFF'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    da_post = ds_post['OUT_RUNOFF'].sel(time=slice(plot_start_time, plot_end_time))
    ts_post_mean = da_post.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_RUNOFF'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate post_mean vs. truth
    df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
    rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.post.runoff.html'.format(lat, lon)))
    
    p = figure(title='Surface runoff, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="Runoff (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Post-processed, mean RMSE={:.3f} mm".format(rmse_post_mean)
        else:
            legend=False
        ts = da_post.sel(N=i+1).to_series()
        p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
    # plot truth
    ts = ts_truth
    p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
    # plot open-loop
    ts = ts_openloop
    p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
           legend="Open-loop, RMSE={:.3f} mm".format(rmse_openloop), line_width=2)
    # Save
    save(p)

# ========================================================== #
# Plot - baseflow
# ========================================================== #
if not linear_model:
    print('\tPlot - baseflow...')
    # --- RMSE --- #
    # extract time series
    ts_truth = ds_truth['OUT_BASEFLOW'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    da_post = ds_post['OUT_BASEFLOW'].sel(time=slice(plot_start_time, plot_end_time))
    ts_post_mean = da_post.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_BASEFLOW'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate post_mean vs. truth
    df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
    rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Regular plots ----- #
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    # plot each ensemble member
    for i in range(N):
        if i == 0:
            legend=True
        else:
            legend=False
        da_post.sel(N=i+1).to_series().plot(
                    color='blue', style='-', alpha=0.3,
                    label='Post-processed, mean RMSE={:.3f} mm'.format(rmse_post_mean),
                    legend=legend)
    # plot truth
    ts_truth.plot(color='k', style='-', label='Truth', legend=True)
    # plot open-loop
    ts_openloop.plot(color='m', style='--',
                     label='Open-loop, RMSE={:.3f} mm'.format(rmse_openloop),
                     legend=True)
    # Make plot looks better
    plt.xlabel('Time')
    plt.ylabel('Baseflow (mm)')
    plt.title('Baseflow, {}, {}, N={}'.format(lat, lon, N))
    # Save figure
    fig.savefig(os.path.join(output_dir, '{}_{}.post.baseflow.png'.format(lat, lon)),
                format='png')

# ========================================================== #
# Plot - total runoff
# ========================================================== #
if not linear_model:
    print('\tPlot - total runoff...')
    # --- RMSE --- #
    # extract time series
    ts_truth = (ds_truth['OUT_RUNOFF'] + ds_truth['OUT_BASEFLOW']).sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    da_post = (ds_post['OUT_RUNOFF'] + ds_post['OUT_BASEFLOW']).sel(
                    time=slice(plot_start_time, plot_end_time))
    ts_post_mean = da_post.mean(dim='N').\
                   to_series()
    ts_openloop = (ds_openloop['OUT_RUNOFF'] + ds_openloop['OUT_BASEFLOW']).sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate post_mean vs. truth
    df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
    rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.post.total_runoff.html'.format(lat, lon)))
    
    p = figure(title='Total runoff, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="Total runoff (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Post-processed, mean RMSE={:.3f} mm".format(rmse_post_mean)
        else:
            legend=False
        ts = da_post.sel(N=i+1).to_series()
        p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
    # plot truth
    ts = ts_truth
    p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
    # plot open-loop
    ts = ts_openloop
    p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
           legend="Open-loop, RMSE={:.3f} mm".format(rmse_openloop), line_width=2)
    # Save
    save(p)

# ========================================================== #
# Plot - SWE
# ========================================================== #
if not linear_model:
    print('\tPlot - SWE...')
    # --- RMSE --- #
    # extract time series
    ts_truth = ds_truth['OUT_SWE'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    da_post = ds_post['OUT_SWE'].sel(time=slice(plot_start_time, plot_end_time))
    ts_post_mean = da_post.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_SWE'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate post_mean vs. truth
    df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
    rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.post.swe.html'.format(lat, lon)))
    
    p = figure(title='Surface SWE, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="SWE (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Post-processed, mean RMSE={:.3f} mm".format(rmse_post_mean)
        else:
            legend=False
        ts = da_post.sel(N=i+1).to_series()
        p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
    # plot truth
    ts = ts_truth
    p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
    # plot open-loop
    ts = ts_openloop
    p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
           legend="Open-loop, RMSE={:.3f} mm".format(rmse_openloop), line_width=2)
    # Save
    save(p)

# ========================================================== #
# Plot - evap
# ========================================================== #
if not linear_model:
    print('\tPlot - EVAP...')
    # --- RMSE --- #
    # extract time series
    ts_truth = ds_truth['OUT_EVAP'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    da_post = ds_post['OUT_EVAP'].sel(time=slice(plot_start_time, plot_end_time))
    ts_post_mean = da_post.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_EVAP'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate post_mean vs. truth
    df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
    rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.post.evap.html'.format(lat, lon)))
    
    p = figure(title='ET, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="ET (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Post-processed, mean RMSE={:.3f} mm".format(rmse_post_mean)
        else:
            legend=False
        ts = da_post.sel(N=i+1).to_series()
        p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
    # plot truth
    ts = ts_truth
    p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
    # plot open-loop
    ts = ts_openloop
    p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
           legend="Open-loop, RMSE={:.3f} mm".format(rmse_openloop), line_width=2)
    # Save
    save(p)

# ========================================================== #
# Plot - snowmelt
# ========================================================== #
if not linear_model:
    print('\tPlot - snowmelt...')
    # --- RMSE --- #
    # extract time series
    ts_truth = ds_truth['OUT_SNOW_MELT'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    da_post = ds_post['OUT_SNOW_MELT'].sel(time=slice(plot_start_time, plot_end_time))
    ts_post_mean = da_post.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_SNOW_MELT'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate post_mean vs. truth
    df_truth_post = pd.concat([ts_truth, ts_post_mean], axis=1, keys=['truth', 'post_mean']).dropna()
    rmse_post_mean = rmse(df_truth_post['truth'], df_truth_post['post_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.post.snowmelt.html'.format(lat, lon)))
    
    p = figure(title='Snowmelt, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="ET (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Post-processed, mean RMSE={:.3f} mm".format(rmse_post_mean)
        else:
            legend=False
        ts = da_post.sel(N=i+1).to_series()
        p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
    # plot truth
    ts = ts_truth
    p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
    # plot open-loop
    ts = ts_openloop
    p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
           legend="Open-loop, RMSE={:.3f} mm".format(rmse_openloop), line_width=2)
    # Save
    save(p)
