
# Usage:
#   python analyze.py <config_file> <debug>

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
import timeit


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

# --- Number of processors --- #
nproc = int(sys.argv[2])

# ========================================================== #
# Parameter setting
# ========================================================== #

if 'LINEAR_MODEL' in cfg:
    linear_model = True
else:
    linear_model = False

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
output_rootdir = cfg['POSTPROCESS']['output_post_dir']


# ========================================================== #
# Setup output data dir
# ========================================================== #
output_dir = setup_output_dirs(
                    output_rootdir,
                    mkdirs=['maps'])['maps']


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

# --- Postprocess results --- #
print('\tPostprocess results...')
ds_post_mean = xr.open_dataset(os.path.join(
                    output_rootdir, 'data', 'ens_mean',
                    'ens_mean.concat.{}_{}.nc'.format(start_year, end_year)))


# ======================================================== #
# Extract shared coordinates
# ======================================================== #
lat_coord = ds_openloop['lat']
lon_coord = ds_openloop['lon']


# ======================================================== #
# Extract soil layer depths
# ======================================================== #
da_soil_depth = get_soil_depth(vic_param_nc)  # [nlayer, lat, lon]
depth_sm1 = da_soil_depth.sel(nlayer=0)  # [lat, lon]
depth_sm2 = da_soil_depth.sel(nlayer=1)  # [lat, lon]
depth_sm3 = da_soil_depth.sel(nlayer=2)  # [lat, lon]


# ======================================================== #
# Plot error map - sm1
# ======================================================== #

# --- Extract variables --- #
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0) / depth_sm1
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0) / depth_sm1
da_post_mean = ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=0) / depth_sm1

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
# Save RMSE for later use
da_rmse_openloop_sm1 = da_rmse_openloop.copy()  # [mm/mm]
da_rmse_post_mean_sm1 = da_rmse_post_mean.copy()
# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.07)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('sm1, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm1_openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.07)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('sm1, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm1_post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr', vmin=-0.07, vmax=0.07)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm/mm)', fontsize=20)
plt.title('sm1, RMSE diff. (post mean - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm1_diff_post_mean_openloop.png'),
            format='png')

# Fraction of improved error cmp. openloop error - (openloop - post mean) / openloop
fig = plt.figure(figsize=(14, 7))
cs = ((da_rmse_openloop-da_rmse_post_mean)/da_rmse_openloop).plot(
            add_colorbar=False, cmap='gnuplot_r', vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('sm1, fraction of improved RMSE error cmp. openloop error\n' \
          '(openloop - post mean) / openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improved_fraction.sm1.png'),
            format='png')


# ======================================================== #
# Plot error map - sm2
# ======================================================== #
# --- Extract variables --- #
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=1) / depth_sm2
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=1) / depth_sm2
da_post_mean = ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=1) / depth_sm2

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
# Save RMSE for later use
da_rmse_openloop_sm2 = da_rmse_openloop.copy()  # [mm/mm]
da_rmse_post_mean_sm2 = da_rmse_post_mean.copy()
# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.1)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('sm2, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm2_openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.1)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('sm2, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm2_post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr',
            vmin=-0.04, vmax=0.04)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm/mm)', fontsize=20)
plt.title('sm2, RMSE diff. (post mean - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm2_diff_post_mean_openloop.png'),
            format='png')

# Fraction of improved error cmp. openloop error - (openloop - post mean) / openloop
fig = plt.figure(figsize=(14, 7))
cs = ((da_rmse_openloop-da_rmse_post_mean)/da_rmse_openloop).plot(
            add_colorbar=False, cmap='gnuplot_r', vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('sm2, fraction of improved RMSE error cmp. openloop error\n' \
          '(openloop - post mean) / openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improved_fraction.sm2.png'),
            format='png')


# ======================================================== #
# Plot error map - sm3
# ======================================================== #
# --- Extract variables --- #
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=2) / depth_sm3
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=2) / depth_sm3
da_post_mean = ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=2) / depth_sm3

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
# Save RMSE for later use
da_rmse_openloop_sm3 = da_rmse_openloop.copy()  # [mm/mm]
da_rmse_post_mean_sm3 = da_rmse_post_mean.copy()
# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.1)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('sm3, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm3_openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.1)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('sm3, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm3_post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr',
            vmin=-0.08, vmax=0.08)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm/mm)', fontsize=20)
plt.title('sm3, RMSE diff. (post. mean - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_sm3_diff_post_mean_openloop.png'),
            format='png')

# Fraction of improved error cmp. openloop error - (openloop - post mean) / openloop
fig = plt.figure(figsize=(14, 7))
cs = ((da_rmse_openloop-da_rmse_post_mean)/da_rmse_openloop).plot(
            add_colorbar=False, cmap='gnuplot_r', vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('sm3, fraction of improved RMSE error cmp. openloop error\n' \
          '(openloop - post mean) / openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improved_fraction.sm3.png'),
            format='png')


# ======================================================== #
# Plot error map - surface runoff
# ======================================================== #

# --- Extract variables --- #
da_truth = ds_truth['OUT_RUNOFF']
da_openloop = ds_openloop['OUT_RUNOFF']
da_post_mean = ds_post_mean['OUT_RUNOFF']

# --- Calculate RMSE --- #
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]

# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]

# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Surface runoff, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff.rmse.openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Surface runoff, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff.rmse.post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr',
            vmin=-0.03, vmax=0.03)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm)', fontsize=20)
plt.title('Surface runoff, RMSE diff. (post mean - openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff.rmse_diff.post_mean_openloop.png'),
            format='png')


# ======================================================== #
# Plot error map - surface runoff, daily
# ======================================================== #
print('Plotting surface runoff, daily...')

# --- Extract variables --- #
time1 = timeit.default_timer()
da_truth = ds_truth['OUT_RUNOFF'].resample('1D', dim='time', how='sum')
da_openloop = ds_openloop['OUT_RUNOFF'].resample('1D', dim='time', how='sum')
da_post_mean = ds_post_mean['OUT_RUNOFF'].resample('1D', dim='time', how='sum')
time2 = timeit.default_timer()
print('part 1 time: {}'.format(time2-time1))

# --- Calculate RMSE --- #
time1 = timeit.default_timer()
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
time2 = timeit.default_timer()
print('part 2 time: {}'.format(time2-time1))

time1 = timeit.default_timer()
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
time2 = timeit.default_timer()
print('part 3 time: {}'.format(time2-time1))

time1 = timeit.default_timer()
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
time2 = timeit.default_timer()
print('part 4 time: {}'.format(time2-time1))

# Save RMSE for later use
da_rmse_openloop_runoff_daily = da_rmse_openloop.copy()  # [mm/day]
da_rmse_post_mean_runoff_daily = da_rmse_post_mean.copy()

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=3.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Surface runoff daily, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff_daily.rmse.openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=3.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Surface runoff daily, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff_daily.rmse.post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr',
            vmin=-0.12, vmax=0.12)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm)', fontsize=20)
plt.title('Surface runoff daily, RMSE diff. (post mean - openloop, both wrt. truth)',
          fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff_daily.rmse_diff.post_mean_openloop.png'),
            format='png')

# Fraction of improved error cmp. openloop error - (openloop - post mean) / openloop
fig = plt.figure(figsize=(14, 7))
cs = ((da_rmse_openloop-da_rmse_post_mean)/da_rmse_openloop).plot(
            add_colorbar=False, cmap='gnuplot_r', vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily surface runoff, fraction of improved RMSE error cmp. openloop error\n' \
          '(openloop - post mean) / openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improved_fraction.runoff_daily.png'),
            format='png')


# ======================================================== #
# Plot error map - surface runoff, weekly
# ======================================================== #
print('Plotting surface runoff, weekly...')

# --- Extract variables --- #
time1 = timeit.default_timer()
da_truth = ds_truth['OUT_RUNOFF'].resample('7D', dim='time', how='sum')
da_openloop = ds_openloop['OUT_RUNOFF'].resample('7D', dim='time', how='sum')
da_post_mean = ds_post_mean['OUT_RUNOFF'].resample('7D', dim='time', how='sum')
time2 = timeit.default_timer()
print('part 1 time: {}'.format(time2-time1))

# --- Calculate RMSE --- #
time1 = timeit.default_timer()
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
time2 = timeit.default_timer()
print('part 2 time: {}'.format(time2-time1))

time1 = timeit.default_timer()
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
time2 = timeit.default_timer()
print('part 3 time: {}'.format(time2-time1))

time1 = timeit.default_timer()
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
time2 = timeit.default_timer()
print('part 4 time: {}'.format(time2-time1))

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=25)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Surface runoff weekly, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff_weekly.rmse.openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=25)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Surface runoff weekly, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff_weekly.rmse.post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr',
            vmin=-0.8, vmax=0.8)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm)', fontsize=20)
plt.title('Surface runoff weekly, RMSE diff. (post mean - openloop, both wrt. truth)',
          fontsize=20)
fig.savefig(os.path.join(output_dir, 'runoff_weekly.rmse_diff.post_mean_openloop.png'),
            format='png')

# ======================================================== #
# Plot error map - baseflow, daily
# ======================================================== #
print('Plotting baseflow, daily...')

# --- Extract variables --- #
time1 = timeit.default_timer()
da_truth = ds_truth['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
da_openloop = ds_openloop['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
da_post_mean = ds_post_mean['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
time2 = timeit.default_timer()
print('part 1 time: {}'.format(time2-time1))

# --- Calculate RMSE --- #
time1 = timeit.default_timer()
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
time2 = timeit.default_timer()
print('part 2 time: {}'.format(time2-time1))

time1 = timeit.default_timer()
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
time2 = timeit.default_timer()
print('part 3 time: {}'.format(time2-time1))

time1 = timeit.default_timer()
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
time2 = timeit.default_timer()
print('part 4 time: {}'.format(time2-time1))

# Save RMSE for later use
da_rmse_openloop_baseflow_daily = da_rmse_openloop.copy()  # [mm/day]
da_rmse_post_mean_baseflow_daily = da_rmse_post_mean.copy()

# --- Plot maps --- #
# Openloop
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop.plot(add_colorbar=False, cmap='viridis',
                           vmin=0, vmax=0.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Baseflow daily, RMSE of openloop (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'baseflow_daily.rmse.openloop.png'), format='png')

# post_mean
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_post_mean.plot(add_colorbar=False, cmap='viridis',
                            vmin=0, vmax=0.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm)', fontsize=20)
plt.title('Baseflow daily, RMSE of postprocessed mean (wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'baseflow_daily.rmse.post_mean.png'), format='png')

# Diff - (post mean - openloop)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_post_mean - da_rmse_openloop).plot(
            add_colorbar=False, cmap='bwr',
            vmin=-0.5, vmax=0.5)
cbar = plt.colorbar(cs, extend='both').set_label('RMSE difference (mm)', fontsize=20)
plt.title('Baseflow daily, RMSE diff. (post mean - openloop, both wrt. truth)',
          fontsize=20)
fig.savefig(os.path.join(output_dir, 'baseflow_daily.rmse_diff.post_mean_openloop.png'),
            format='png')

# Fraction of improved error cmp. openloop error - (openloop - post mean) / openloop
fig = plt.figure(figsize=(14, 7))
cs = ((da_rmse_openloop-da_rmse_post_mean)/da_rmse_openloop).plot(
            add_colorbar=False, cmap='gnuplot_r', vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily baseflow, fraction of improved RMSE error cmp. openloop error\n' \
          '(openloop - post mean) / openloop, both wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improved_fraction.baseflow_daily.png'),
            format='png')


# ======================================================== #
# Plot sm/runoff RMSE improvement fraction
# ======================================================== #
# --- surface runoff/sm1 --- #
print('Plotting RMSE improvement fraction, surface runoff daily/sm1...')
da_frac = (da_rmse_post_mean_runoff_daily - da_rmse_openloop_runoff_daily) / \
          ((da_rmse_post_mean_sm1 - da_rmse_openloop_sm1) * depth_sm1)

fig = plt.figure(figsize=(14, 7))
cs = da_frac.plot(add_colorbar=False, cmap='viridis',
                  vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily surface runoff RMSE improvement [mm] / sm1 RMSE improvement [mm]\n'\
          '(baseline: openloop; RMSE calculated wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improv_frac.runoff_daily_sm1.png'),
                         format='png')

# --- surface runoff/sm2 --- #
print('Plotting RMSE improvement fraction, surface runoff daily/sm2...')
da_frac = (da_rmse_post_mean_runoff_daily - da_rmse_openloop_runoff_daily) / \
          ((da_rmse_post_mean_sm2 - da_rmse_openloop_sm2) * depth_sm2)

fig = plt.figure(figsize=(14, 7))
cs = da_frac.plot(add_colorbar=False, cmap='viridis',
                  vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily surface runoff RMSE improvement [mm] / sm2 RMSE improvement [mm]\n'\
          '(baseline: openloop; RMSE calculated wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improv_frac.runoff_daily_sm2.png'),
                         format='png')

# --- surface runoff/(sm1+sm2) --- #
# (1) Calculate RMSE of (sm1+sm2)
# Extract variables
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0) + \
           ds_truth['OUT_SOIL_MOIST'].sel(nlayer=1)
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0) + \
              ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=1)
da_post_mean = ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=0) + \
               ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=1)
# Calculate RMSE
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_sm12 = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean_sm12 = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
# (2) Plot RMSE improvement fraction
print('Plotting RMSE improvement fraction, surface runoff daily/(sm1+sm2)...')
da_frac = (da_rmse_post_mean_runoff_daily - da_rmse_openloop_runoff_daily) / \
          (da_rmse_post_mean_sm12 - da_rmse_openloop_sm12)

fig = plt.figure(figsize=(14, 7))
cs = da_frac.plot(add_colorbar=False, cmap='viridis',
                  vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily surface runoff RMSE improvement [mm] / ' \
          '(sm1+sm2) RMSE improvement [mm]\n'\
          '(baseline: openloop; RMSE calculated wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improv_frac.runoff_daily_sm12.png'),
                         format='png')

# --- baseflow/sm3 --- #
print('Plotting RMSE improvement fraction, baseflow daily/sm3...')
da_frac = (da_rmse_post_mean_baseflow_daily - da_rmse_openloop_baseflow_daily) / \
          ((da_rmse_post_mean_sm3 - da_rmse_openloop_sm3) * depth_sm3)

fig = plt.figure(figsize=(14, 7))
cs = da_frac.plot(add_colorbar=False, cmap='viridis',
                  vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily baseflow RMSE improvement [mm] / sm3 RMSE improvement [mm]\n'\
          '(baseline: openloop; RMSE calculated wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improv_frac.baseflow_daily_sm3.png'),
                         format='png')

# --- Total runoff/ total sm --- #
# (1) Calculate RMSE of (sm1+sm2+sm3)
# Extract variables
da_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0) + \
           ds_truth['OUT_SOIL_MOIST'].sel(nlayer=1) + \
           ds_truth['OUT_SOIL_MOIST'].sel(nlayer=2)
da_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0) + \
              ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=1) + \
              ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=2)
da_post_mean = ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=0) + \
               ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=1) + \
               ds_post_mean['OUT_SOIL_MOIST'].sel(nlayer=2)
# Calculate RMSE
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_smTot = xr.DataArray(rmse_openloop, coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean_smTot = xr.DataArray(rmse_post_mean, coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])

# (2) Calculate RMSE of (surface runoff + baseflow), daily
# Extract variables
da_truth = (ds_truth['OUT_RUNOFF'] + ds_truth['OUT_BASEFLOW']).resample(
                '1D', dim='time', how='sum')
da_openloop = (ds_openloop['OUT_RUNOFF'] + ds_openloop['OUT_BASEFLOW']).resample(
                '1D', dim='time', how='sum')
da_post_mean = (ds_post_mean['OUT_RUNOFF'] + ds_post_mean['OUT_BASEFLOW']).resample(
                '1D', dim='time', how='sum')
# Calculate RMSE
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_truth.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
openloop = da_openloop.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
post_mean = da_post_mean.values.reshape([len(da_openloop['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_openloop = np.array(list(map(
            lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_post_mean = np.array(list(map(
            lambda j: rmse(truth[:, j], post_mean[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_post_mean = rmse_post_mean.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_runoffTot_daily = xr.DataArray(rmse_openloop,
                                                coords=[lat_coord, lon_coord],
                                dims=['lat', 'lon'])
da_rmse_post_mean_runoffTot_daily = xr.DataArray(rmse_post_mean,
                                                 coords=[lat_coord, lon_coord],
                                 dims=['lat', 'lon'])
# (2) Plot RMSE improvement fraction
print('Plotting RMSE improvement fraction, total runoff daily/total sm...')
da_frac = (da_rmse_post_mean_runoffTot_daily - da_rmse_openloop_runoffTot_daily) / \
          (da_rmse_post_mean_smTot - da_rmse_openloop_smTot)

fig = plt.figure(figsize=(14, 7))
cs = da_frac.plot(add_colorbar=False, cmap='viridis',
                  vmin=0, vmax=1)
cbar = plt.colorbar(cs, extend='both').set_label('Fraction', fontsize=20)
plt.title('Daily total runoff RMSE improvement [mm] / ' \
          'total sm RMSE improvement [mm]\n'\
          '(baseline: openloop; RMSE calculated wrt. truth)', fontsize=20)
fig.savefig(os.path.join(output_dir, 'rmse_improv_frac.runoffTot_daily_smTot.png'),
                         format='png')

