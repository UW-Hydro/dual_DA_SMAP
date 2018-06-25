
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
import multiprocessing as mp
from collections import OrderedDict

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


def load_nc_file_cell(nc_file, start_year, end_year,
                      lat, lon):
    ''' Loads in nc files for all years.

    Parameters
    ----------
    nc_file: <str>
        netCDF file to load, with {} to be substituted by YYYY
    start_year: <int>
        Start year
    end_year: <int>
        End year
    lat: <float>
        lat of grid cell to extract
    lon: <float>
        lon of grid cell to extract

    Returns
    ----------
    ds_all_years: <xr.Dataset>
        Dataset of all years
    '''

    list_ds = []
    for year in range(start_year, end_year+1):
        # Load data
        fname = nc_file.format(year)
        ds = xr.open_dataset(fname).sel(lat=lat, lon=lon)
        list_ds.append(ds)
    # Concat all years
    ds_all_years = xr.concat(list_ds, dim='time')

    return ds_all_years


def setup_output_dirs(out_basedir, mkdirs=['results', 'state',
                                            'logs', 'plots']):
    ''' This function creates output directories.

    Parameters
    ----------
    out_basedir: <str>
        Output base directory for all output files
    mkdirs: <list>
        A list of subdirectories to make

    Require
    ----------
    os
    OrderedDict

    Returns
    ----------
    dirs: <OrderedDict>
        A dictionary of subdirectories

    '''

    dirs = OrderedDict()
    for d in mkdirs:
        dirs[d] = os.path.join(out_basedir, d)

    for dirname in dirs.values():
        os.makedirs(dirname, exist_ok=True)

    return dirs


# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.argv[1])

# --- lat and lon --- #
lat = float(sys.argv[2])
lon = float(sys.argv[3])

# ========================================================== #
# Parameter setting
# ========================================================== #
# --- Input directory and files --- #
# gen_synthetic results
gen_synth_basedir = cfg['SYNTHETIC']['gen_synth_basedir']
truth_nc_filename = cfg['SYNTHETIC']['truth_nc_filename']
synth_meas_nc_filename = cfg['SYNTHETIC']['synth_meas_nc_filename']

# openloop
openloop_nc = cfg['SYNTHETIC']['openloop_nc']

# initial time
init_time = pd.to_datetime(cfg['SYNTHETIC']['init_time'])

# VIC global file template (for extracting param file and snow_band)
vic_global_txt = cfg['SYNTHETIC']['vic_global_txt']

# Forcings (for all basepaths, 'YYYY.nc' will be appended)
orig_force_basepath = cfg['SYNTHETIC']['orig_force_basepath']
truth_force_basepath = cfg['SYNTHETIC']['truth_force_basepath']
# ens_force_basedir/ens_<i>/force.<YYYY>.nc, where <i> = 1, 2, ..., N
ens_force_basedir = cfg['SYNTHETIC']['ens_force_basedir']

# VIC parameter netCDF file
vic_param_nc = cfg['SYNTHETIC']['vic_param_nc']

# --- Measurement times --- #
times = pd.date_range(cfg['SYNTHETIC']['meas_start_time'],
                      cfg['SYNTHETIC']['meas_end_time'],
                      freq=cfg['SYNTHETIC']['freq'])
ntime = len(times)

# --- Plot time period --- #
plot_start_time = pd.to_datetime(cfg['SYNTHETIC']['plot_start_time'])
plot_end_time = pd.to_datetime(cfg['SYNTHETIC']['plot_end_time'])
start_year = plot_start_time.year
end_year = plot_end_time.year

# --- Output --- #
output_rootdir = cfg['OUTPUT']['output_dir']


# ========================================================== #
# Setup output data dir
# ========================================================== #
dirname = 'time_series.{}_{}'.format(lat, lon)
output_dir = setup_output_dirs(
                    output_rootdir,
                    mkdirs=[dirname])[dirname]


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

# --- Truth states + orig. precip --- #
print('\ttruthStates_origP...')
nc_files = os.path.join(gen_synth_basedir,
                        'test.truth_states_orig_forcing',
                        'history',
                        'history.concat.{}.nc')
ds_truthState_origP = load_nc_file_cell(nc_files, start_year, end_year,
                                 lat, lon)


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
ts_truthState_origP = ds_truthState_origP['OUT_SOIL_MOIST'].sel(
                            nlayer=0,
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series() / depth_sm1
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=0,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
rmse_meas = rmse(df_truth_meas['truth'].values, df_truth_meas['meas'])
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truthState_origP
ts_truthState_origP.plot(
                color='blue', style='-', alpha=0.3,
                label='truthState_origP, mean RMSE={:.3f} mm/mm'.format(rmse_truthState_origP),
                legend=True)
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
plt.title('Top-layer soil moisture, {}, {}'.format(lat, lon))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.truthState_origP.sm1.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.sm1.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm/mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
ts_truthState_origP = ds_truthState_origP['OUT_SOIL_MOIST'].sel(
                            nlayer=1,
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series() / depth_sm2
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=1,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm2
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truthState_origP
ts_truthState_origP.plot(
    color='blue', style='-', alpha=0.3,
    label='truthState_origP, mean RMSE={:.5f} mm/mm'.format(rmse_truthState_origP),
    legend=True)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.5f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Middle-layer soil moisture, {}, {}'.format(lat, lon))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.truthState_origP.sm2.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.sm2.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm/mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
ts_truthState_origP = ds_truthState_origP['OUT_SOIL_MOIST'].sel(
                            nlayer=2,
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series() / depth_sm3
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=2,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm3
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truthState_origP
ts_truthState_origP.plot(
    color='blue', style='-', alpha=0.3,
    label='truthState_origP, mean RMSE={:.5f} mm/mm'.format(rmse_truthState_origP),
    legend=True)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.5f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Bottom-layer soil moisture, {}, {}'.format(lat, lon))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.truthState_origP.sm3.png'.format(lat, lon)),
            format='png')

# ========================================================== #
# Plot - runoff, subdaily
# ========================================================== #
print('\tPlot - surface runoff...')
# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_RUNOFF'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
ts_truthState_origP = ds_truthState_origP['OUT_RUNOFF'].sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series()
ts_openloop = ds_openloop['OUT_RUNOFF'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.runoff.html'.format(lat, lon)))

p = figure(title='Surface runoff, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Runoff (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
# Plot - runoff, aggregated to daily
# ========================================================== #
print('\tPlot - surface runoff, aggregated to daily...')
# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_RUNOFF'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series().\
           resample("D", how='sum')
ts_truthState_origP = ds_truthState_origP['OUT_RUNOFF'].sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series().resample("D", how='sum')
ts_openloop = ds_openloop['OUT_RUNOFF'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series().\
              resample("D", how='sum')
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.runoff_daily.html'.format(lat, lon)))

p = figure(title='Surface runoff, daily, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Runoff (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
print('\tPlot - baseflow...')
# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_BASEFLOW'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
ts_truthState_origP = ds_truthState_origP['OUT_BASEFLOW'].sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series()
ts_openloop = ds_openloop['OUT_BASEFLOW'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Regular plots ----- #
# Create figure
fig = plt.figure(figsize=(12, 6))
# plot truthState_origP
ts_truthState_origP.plot(
    color='blue', style='-', alpha=0.3,
    label='truthState_origP, mean RMSE={:.3f} mm'.format(rmse_truthState_origP),
    legend=True)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.3f} mm'.format(rmse_openloop),
                 legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Baseflow (mm)')
plt.title('Baseflow, {}, {}'.format(lat, lon))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.truthState_origP.baseflow.png'.format(lat, lon)),
            format='png')

# ========================================================== #
# Plot - total runoff
# ========================================================== #
print('\tPlot - total runoff...')
# --- RMSE --- #
# extract time series
ts_truth = (ds_truth['OUT_RUNOFF'] + ds_truth['OUT_BASEFLOW']).sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
ts_truthState_origP = (ds_truthState_origP['OUT_RUNOFF'] + \
                       ds_truthState_origP['OUT_BASEFLOW']).sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series()
ts_openloop = (ds_openloop['OUT_RUNOFF'] + ds_openloop['OUT_BASEFLOW']).sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.total_runoff.html'.format(lat, lon)))

p = figure(title='Total runoff, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="Total runoff (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
print('\tPlot - SWE...')
# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_SWE'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
ts_truthState_origP = ds_truthState_origP['OUT_SWE'].sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series()
ts_openloop = ds_openloop['OUT_SWE'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.swe.html'.format(lat, lon)))

p = figure(title='Surface SWE, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="SWE (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.3f} mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
print('\tPlot - EVAP...')
# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_EVAP'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
ts_truthState_origP = ds_truthState_origP['OUT_EVAP'].sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series()
ts_openloop = ds_openloop['OUT_EVAP'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.evap.html'.format(lat, lon)))

p = figure(title='ET, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="ET (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
print('\tPlot - snowmelt...')
# --- RMSE --- #
# extract time series
ts_truth = ds_truth['OUT_SNOW_MELT'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
ts_truthState_origP = ds_truthState_origP['OUT_SNOW_MELT'].sel(
                            time=slice(plot_start_time, plot_end_time)).\
                      to_series()
ts_openloop = ds_openloop['OUT_SNOW_MELT'].sel(
                lat=lat, lon=lon,
                time=slice(plot_start_time, plot_end_time)).to_series()
# Calculate truthState_origP vs. truth
df_truth_truthState_origP = pd.concat([ts_truth, ts_truthState_origP], axis=1, keys=['truth', 'truthState_origP']).dropna()
rmse_truthState_origP = rmse(df_truth_truthState_origP['truth'], df_truth_truthState_origP['truthState_origP'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.truthState_origP.snowmelt.html'.format(lat, lon)))

p = figure(title='Snowmelt, {}, {}'.format(lat, lon),
           x_axis_label="Time", y_axis_label="ET (mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot truthState_origP
legend="truthState_origP, mean RMSE={:.2f} mm".format(rmse_truthState_origP)
ts = ts_truthState_origP
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
