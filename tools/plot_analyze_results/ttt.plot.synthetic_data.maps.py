
# Usage:
#   python analyze.py <synthetic_data_config_file>

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
    ds_soil = xr.open_dataset(param_nc)

    # Get soil depth for each layer
    # Dimension: [nlayer, lat, lon]
    da_soil_depth = ds_soil['depth']  # [m]
    # Convert unit to mm
    da_soil_depth = da_soil_depth * 1000 # [mm]

    return da_soil_depth


def load_nc_file(nc_file, start_year, end_year):
    ''' Loads in nc files for all years.

    Parameters
    ----------
    nc_file: <str>
        netCDF file to load, with {} to be substituted by YYYY
    start_year: <int>
        Start year
    end_year: <int>
        End year

    Returns
    ----------
    ds_all_years: <xr.Dataset>
        Dataset of all years
    '''

    list_ds = []
    for year in range(start_year, end_year+1):
        # Load data
        fname = nc_file.format(year)
        ds = xr.open_dataset(fname)
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


def calc_sm_runoff_corrcoef(sm, runoff):
    runoff_with_runoff = runoff[runoff>0]
    sm_with_runoff = sm[runoff>0]
    
    return np.corrcoef(sm_with_runoff, runoff_with_runoff)[0, 1]


def to_netcdf_history_file_compress(ds_hist, out_nc):
    ''' This function saves a VIC-history-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_hist: <xr.Dataset>
        History dataset to save
    out_nc: <str>
        Path of output netCDF file
    '''

    dict_encode = {}
    for var in ds_hist.data_vars:
        # skip variables not starting with "OUT_"
        if var.split('_')[0] != 'OUT':
            continue
        # determine chunksizes
        chunksizes = []
        for i, dim in enumerate(ds_hist[var].dims):
            if dim == 'time':  # for time dimension, chunksize = 1
                chunksizes.append(1)
            else:
                chunksizes.append(len(ds_hist[dim]))
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1,
                            'chunksizes': chunksizes}
    ds_hist.to_netcdf(out_nc,
                      format='NETCDF4',
                      encoding=dict_encode)

# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.argv[1])


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

# --- Truth states + orig. precip --- #
print('\ttruthStates_origP...')
#nc_files = os.path.join(gen_synth_basedir,
#                        'test.truth_states_orig_forcing',
#                        'history',
#                        'history.concat.{}.nc')
#ds_truthState_origP = load_nc_file(nc_files, start_year, end_year)
#to_netcdf_history_file_compress(
#        ds_truthState_origP,
#        os.path.join(gen_synth_basedir,
#                     'test.truth_states_orig_forcing',
#                     'history',
#                     'history.concat.19800101_19891231.nc'))
ds_truthState_origP = xr.open_dataset(os.path.join(
            gen_synth_basedir,
            'test.truth_states_orig_forcing',
            'history',
            'history.concat.19800101_19891231.nc'))


# ======================================================== #
# Extract shared coordinates
# ======================================================== #
lat_coord = ds_openloop['lat']
lon_coord = ds_openloop['lon']


# ======================================================== #
# Extract variables
# ======================================================== #
import pickle
print('Extracting variables...')
# Extract openloop variables
print('sm1...')
da_openloop_sm1 = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0)
da_truth_sm1 = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0)
da_truthState_origP_sm1 = ds_truthState_origP['OUT_SOIL_MOIST'].sel(nlayer=0)
with open('./vars/da_openloop_sm1', 'wb') as f:
    pickle.dump(da_openloop_sm1, f)
with open('./vars/da_truth_sm1', 'wb') as f:
    pickle.dump(da_truth_sm1, f)
with open('./vars/da_truthState_origP_sm1', 'wb') as f:
    pickle.dump(da_truthState_origP_sm1, f)

print('sm2...')
da_openloop_sm2 = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=1)
da_truth_sm2 = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=1)
da_truthState_origP_sm2 = ds_truthState_origP['OUT_SOIL_MOIST'].sel(nlayer=1)
with open('./vars/da_openloop_sm2', 'wb') as f:
    pickle.dump(da_openloop_sm2, f)
with open('./vars/da_truth_sm2', 'wb') as f:
    pickle.dump(da_truth_sm2, f)
with open('./vars/da_truthState_origP_sm2', 'wb') as f:
    pickle.dump(da_truthState_origP_sm2, f)

print('sm3...')
da_openloop_sm3 = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=2)
da_truth_sm3 = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=2)
da_truthState_origP_sm3 = ds_truthState_origP['OUT_SOIL_MOIST'].sel(nlayer=2)
with open('./vars/da_openloop_sm3', 'wb') as f:
    pickle.dump(da_openloop_sm3, f)
with open('./vars/da_truth_sm3', 'wb') as f:
    pickle.dump(da_truth_sm3, f)
with open('./vars/da_truthState_origP_sm3', 'wb') as f:
    pickle.dump(da_truthState_origP_sm3, f)

print('runoff...')
da_openloop_runoff = ds_openloop['OUT_RUNOFF'].resample('1D', dim='time', how='sum')
da_truth_runoff = ds_truth['OUT_RUNOFF'].resample('1D', dim='time', how='sum')
da_truthState_origP_runoff = ds_truthState_origP['OUT_RUNOFF'].resample('1D', dim='time', how='sum')
with open('./vars/da_openloop_runoff', 'wb') as f:
    pickle.dump(da_openloop_runoff, f)
with open('./vars/da_truth_runoff', 'wb') as f:
    pickle.dump(da_truth_runoff, f)
with open('./vars/da_truthState_origP_runoff', 'wb') as f:
    pickle.dump(da_truthState_origP_runoff, f)

print('baseflow...')
da_openloop_baseflow = ds_openloop['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
da_truth_baseflow = ds_truth['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
da_truthState_origP_baseflow = ds_truthState_origP['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
with open('./vars/da_openloop_baseflow', 'wb') as f:
    pickle.dump(da_openloop_baseflow, f)
with open('./vars/da_truth_baseflow', 'wb') as f:
    pickle.dump(da_truth_baseflow, f)
with open('./vars/da_truthState_origP_baseflow', 'wb') as f:
    pickle.dump(da_truthState_origP_baseflow, f)

# ======================================================== #
# Calculate RMSE - sm1
# ======================================================== #
print('Calculating RMSE - sm1 ...')
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
print('\tReshaping...')
truth = da_truth_sm1.values.reshape([len(da_openloop_sm1['time']),
                                     nloop])  # [time, nloop]
openloop = da_openloop_sm1.values.reshape([len(da_openloop_sm1['time']),
                                           nloop])  # [time, nloop]
truthState_origP = da_truthState_origP_sm1.values.reshape(
            [len(da_openloop_sm1['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
print('\tCalculating RMSE...')
rmse_openloop = np.array(list(map(
             lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_truthState_origP = np.array(list(map(
            lambda j: rmse(truth[:, j], truthState_origP[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
print('\tFinalizing...')
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_truthState_origP = rmse_truthState_origP.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_sm1 = xr.DataArray(rmse_openloop,
                                    coords=[lat_coord, lon_coord],
                                    dims=['lat', 'lon'])
da_rmse_truthState_origP_sm1 = xr.DataArray(rmse_truthState_origP,
                                            coords=[lat_coord, lon_coord],
                                            dims=['lat', 'lon'])
# Save RMSE for later use
with open('./vars/da_rmse_openloop_sm1', 'wb') as f:
    pickle.dump(da_rmse_openloop_sm1, f)
with open('./vars/da_rmse_truthState_origP_sm1', 'wb') as f:
    pickle.dump(da_rmse_truthState_origP_sm1, f)

# ======================================================== #
# Calculate RMSE - sm2
# ======================================================== #
print('Calculating RMSE - sm2 ...')
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
print('\tReshaping...')
truth = da_truth_sm2.values.reshape([len(da_openloop_sm2['time']),
                                     nloop])  # [time, nloop]
openloop = da_openloop_sm2.values.reshape([len(da_openloop_sm2['time']),
                                           nloop])  # [time, nloop]
truthState_origP = da_truthState_origP_sm2.values.reshape(
            [len(da_openloop_sm2['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
print('\tCalculating RMSE...')
rmse_openloop = np.array(list(map(
             lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_truthState_origP = np.array(list(map(
            lambda j: rmse(truth[:, j], truthState_origP[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
print('\tFinalizing...')
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_truthState_origP = rmse_truthState_origP.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_sm2 = xr.DataArray(rmse_openloop,
                                    coords=[lat_coord, lon_coord],
                                    dims=['lat', 'lon'])
da_rmse_truthState_origP_sm2 = xr.DataArray(rmse_truthState_origP,
                                            coords=[lat_coord, lon_coord],
                                            dims=['lat', 'lon'])
# Save RMSE for later use
with open('./vars/da_rmse_openloop_sm2', 'wb') as f:
    pickle.dump(da_rmse_openloop_sm2, f)
with open('./vars/da_rmse_truthState_origP_sm2', 'wb') as f:
    pickle.dump(da_rmse_truthState_origP_sm2, f)

# ======================================================== #
# Calculate RMSE - sm3
# ======================================================== #
print('Calculating RMSE - sm3 ...')
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
print('\tReshaping...')
truth = da_truth_sm3.values.reshape([len(da_openloop_sm3['time']),
                                     nloop])  # [time, nloop]
openloop = da_openloop_sm3.values.reshape([len(da_openloop_sm3['time']),
                                           nloop])  # [time, nloop]
truthState_origP = da_truthState_origP_sm3.values.reshape(
            [len(da_openloop_sm3['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
print('\tCalculating RMSE...')
rmse_openloop = np.array(list(map(
             lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_truthState_origP = np.array(list(map(
            lambda j: rmse(truth[:, j], truthState_origP[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
print('\tFinalizing...')
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_truthState_origP = rmse_truthState_origP.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_sm3 = xr.DataArray(rmse_openloop,
                                    coords=[lat_coord, lon_coord],
                                    dims=['lat', 'lon'])
da_rmse_truthState_origP_sm3 = xr.DataArray(rmse_truthState_origP,
                                            coords=[lat_coord, lon_coord],
                                            dims=['lat', 'lon'])
# Save RMSE for later use
with open('./vars/da_rmse_openloop_sm3', 'wb') as f:
    pickle.dump(da_rmse_openloop_sm3, f)
with open('./vars/da_rmse_truthState_origP_sm3', 'wb') as f:
    pickle.dump(da_rmse_truthState_origP_sm3, f)

# ======================================================== #
# Calculate RMSE - surface runoff
# ======================================================== #
print('Calculating RMSE - runoff ...')
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
print('\tReshaping...')
truth = da_truth_runoff.values.reshape([len(da_openloop_runoff['time']),
                                     nloop])  # [time, nloop]
openloop = da_openloop_runoff.values.reshape([len(da_openloop_runoff['time']),
                                           nloop])  # [time, nloop]
truthState_origP = da_truthState_origP_runoff.values.reshape(
            [len(da_openloop_runoff['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
print('\tCalculating RMSE...')
rmse_openloop = np.array(list(map(
             lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_truthState_origP = np.array(list(map(
            lambda j: rmse(truth[:, j], truthState_origP[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_truthState_origP = rmse_truthState_origP.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_runoff = xr.DataArray(rmse_openloop,
                                    coords=[lat_coord, lon_coord],
                                    dims=['lat', 'lon'])
da_rmse_truthState_origP_runoff = xr.DataArray(rmse_truthState_origP,
                                            coords=[lat_coord, lon_coord],
                                            dims=['lat', 'lon'])
# Save RMSE for later use
with open('./vars/da_rmse_openloop_runoff', 'wb') as f:
    pickle.dump(da_rmse_openloop_runoff, f)
with open('./vars/da_rmse_truthState_origP_runoff', 'wb') as f:
    pickle.dump(da_rmse_truthState_origP_runoff, f)

# ======================================================== #
# Calculate RMSE - baseflow
# ======================================================== #
print('Calculating RMSE - baseflow ...')
# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
print('\tReshaping...')
truth = da_truth_baseflow.values.reshape([len(da_openloop_baseflow['time']),
                                     nloop])  # [time, nloop]
openloop = da_openloop_baseflow.values.reshape([len(da_openloop_baseflow['time']),
                                           nloop])  # [time, nloop]
truthState_origP = da_truthState_origP_baseflow.values.reshape(
            [len(da_openloop_baseflow['time']), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
print('\tCalculating RMSE...')
rmse_openloop = np.array(list(map(
             lambda j: rmse(truth[:, j], openloop[:, j]),
            range(nloop))))  # [nloop]
rmse_truthState_origP = np.array(list(map(
            lambda j: rmse(truth[:, j], truthState_origP[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
print('\tFinalizing...')
rmse_openloop = rmse_openloop.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_truthState_origP = rmse_truthState_origP.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_openloop_baseflow = xr.DataArray(rmse_openloop,
                                    coords=[lat_coord, lon_coord],
                                    dims=['lat', 'lon'])
da_rmse_truthState_origP_baseflow = xr.DataArray(rmse_truthState_origP,
                                            coords=[lat_coord, lon_coord],
                                            dims=['lat', 'lon'])
# Save RMSE for later use
with open('./vars/da_rmse_openloop_baseflow', 'wb') as f:
    pickle.dump(da_rmse_openloop_baseflow, f)
with open('./vars/da_rmse_truthState_origP_baseflow', 'wb') as f:
    pickle.dump(da_rmse_truthState_origP_baseflow, f)

