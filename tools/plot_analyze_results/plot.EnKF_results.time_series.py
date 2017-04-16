
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


# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.argv[1])

# --- Number of processors --- #
nproc = int(sys.argv[2])

# --- Whether to load and plot debug variables --- #
debug = (sys.argv[3].lower() == 'true')

# --- lat and lon --- #
lat = float(sys.argv[4])
lon = float(sys.argv[5])

# --- Whether to load and plot bias correction variables --- #
bias_correct = (sys.argv[6].lower() == 'true')

# ========================================================== #
# Parameter setting
# ========================================================== #

if 'LINEAR_MODEL' in cfg:
    linear_model = True
else:
    linear_model = False

# --- Input directory and files --- #
# EnKF results
EnKF_result_basedir = cfg['EnKF']['EnKF_result_basedir']

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
output_dir = cfg['OUTPUT']['output_dir']


# ========================================================== #
# Load data
# ========================================================== #
print('Loading data...')
print('\tInnovation...')

# --- Normalized innovation --- #
innov_nc = os.path.join(EnKF_result_basedir, 'temp', 'innov',
                        'innov_norm.concat.{}_{}.nc'.format(
                                start_year, end_year))
da_innov_norm = xr.open_dataset(innov_nc)['innov_norm']
s_innov_norm = da_innov_norm.sel(lat=lat, lon=lon).to_series()

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

# --- EnKF results --- #
print('\tEnKF results...')
if nproc == 1:
    list_ds_EnKF = []
    for i in range(N):
        print('\t\t{}'.format(i+1))
        # Load in results for all years
        nc_file = os.path.join(
                EnKF_result_basedir,
                'history',
                'EnKF_ensemble_concat',
                'history.ens{}.concat.{}.nc'.format(i+1, '{}'))
        ds_all_years = load_nc_file_cell(nc_file, start_year, end_year,
                                         lat, lon)
        # Put data in list
        list_ds_EnKF.append(ds_all_years)
elif nproc > 1:
    list_ds_EnKF = []
    results = {}
    # --- Set up multiprocessing --- #
    pool = mp.Pool(processes=nproc)
    # --- Loop over each ensemble member --- #
    for i in range(N):
        # Load in results for all years
        print('\t\t{}'.format(i+1))
        nc_file = os.path.join(
                EnKF_result_basedir,
                'history',
                'EnKF_ensemble_concat',
                'history.ens{}.concat.{}.nc'.format(i+1, '{}'))
        results[i] = pool.apply_async(load_nc_file_cell,
                                      (nc_file, start_year, end_year,
                                       lat, lon))
    # --- Finish multiprocessing --- #
    pool.close()
    pool.join()
    # --- Get return values --- #
    for i, result in results.items():
        list_ds_EnKF.append(result.get())
# Concat all ensemble members together
print('\t\tConcatenating all ensemble members...')
ds_EnKF = xr.concat(list_ds_EnKF, dim='N')
ds_EnKF['N'] = range(1, N+1)

# --- Bias correction, if specified --- #
if bias_correct:
    print('\tBias correction terms...')
    # --- Load bias correction reference history file --- #
    nc_file = os.path.join(
        EnKF_result_basedir,
        'history',
        'EnKF_ensemble_concat',
        'history.ensref.concat.{}.nc'.format('{}'))
    ds_hist_ref = load_nc_file_cell(nc_file, start_year, end_year, lat, lon)
    # --- Load bias correction delta --- #
    # Get tile fraction
    if not linear_model:
        da_tile_frac = determine_tile_frac(vic_global_txt).sel(lat=lat, lon=lon)  # [veg, snow]
    else:
        da_tile_frac = xr.DataArray([[1]], coords=[[1], [0]], dims=['veg', 'snow'])
    # Load delta
    nc_file = os.path.join(
        EnKF_result_basedir,
        'temp',
        'bias_correct',
        'delta.concat.{}_{}.nc'.format(start_year, end_year))
    ds_bc_delta = xr.open_dataset(nc_file)
    da_bc_delta = ds_bc_delta['delta_soil_moisture'].sel(
        lat=lat, lon=lon)  # [time, veg_class, snow_band, nlayer]
    # ### Calculate cell average values ### #
    # Determine the total number of loops
    nloop = len(da_bc_delta['time']) * len(da_bc_delta['nlayer'])
    # Convert into nloop
    delta = np.rollaxis(da_bc_delta.values, 3, 1)  # [time, nlayer, veg, snow]
    delta = delta.reshape(
            [nloop, len(da_bc_delta['veg_class']), len(da_bc_delta['snow_band'])])  # [nloop, nveg, nsnow]
    tile_frac = da_tile_frac.values  # [nveg, nsnow]
    # Calculate cell-average value
    delta_cellAvg = np.array(list(map(
                lambda i: np.nansum(delta[i, :, :] * tile_frac),
                range(nloop))))  # [nloop]
    # Reshape
    delta_cellAvg = delta_cellAvg.reshape(
            [len(da_bc_delta['time']), len(da_bc_delta['nlayer'])])  # [time, nlayer]
    # Put in da
    da_delta_cellAvg = xr.DataArray(
            delta_cellAvg,
            coords=[da_bc_delta['time'], da_bc_delta['nlayer']],
            dims=['time', 'nlayer'])

# --- Forcings --- #
print('\tForcings...')
# ---- Orig. --- #
print('\t- original')
list_ds_force_orig = []
for year in range(start_year, end_year+1):
    # Load data
    fname = '{}{}.nc'.format(orig_force_basepath, year)
    ds = xr.open_dataset(fname).sel(lat=lat, lon=lon)
    # Put data in list
    list_ds_force_orig.append(ds)
# Concat all years
ds_force_orig = xr.concat(list_ds_force_orig, dim='time')  # [time]
# --- Truth --- #
print('\t- truth')
list_ds_force_truth = []
for year in range(start_year, end_year+1):
    # Load data
    fname = '{}{}.nc'.format(truth_force_basepath, year)
    ds = xr.open_dataset(fname).sel(lat=lat, lon=lon)
    # Put data in list
    list_ds_force_truth.append(ds)
# Concat all years
ds_force_truth = xr.concat(list_ds_force_truth, dim='time')  # [time]
# --- Ensemble members --- #
print('\t- ensemble members')
if nproc == 1:
    list_ds_force_ens = []
    for i in range(N):
        print('\t\t{}'.format(i+1))
        nc_file = '{}/ens_{}/force.{}.nc'.format(ens_force_basedir, i+1, {})
        ds = load_nc_file_cell(nc_file, start_year, end_year, lat, lon)
        # Put data in list
        list_ds_force_ens.append(ds)
elif nproc > 1:
    list_ds_force_ens = []
    results = {}
    # --- Set up multiprocessing --- #
    pool = mp.Pool(processes=nproc)
    # --- Loop over each ensemble member --- #
    for i in range(N):
        print('\t\t{}'.format(i+1))
        nc_file = '{}/ens_{}/force.{}.nc'.format(ens_force_basedir, i+1, {})
        results[i] = pool.apply_async(
                            load_nc_file_cell,
                            (nc_file, start_year, end_year, lat, lon))
    # --- Finish multiprocessing --- #
    pool.close()
    pool.join()
    # --- Get return values --- #
    for i, result in results.items():
        list_ds_force_ens.append(result.get())
# Concat all ensemble members together
ds_force_ens = xr.concat(list_ds_force_ens, dim='N')
ds_force_ens['N'] = range(1, N+1)  # [N, time]

# --- Diagnostics - perturbation --- #
if debug:
    print('\tDiagnostics - perturbation...')
    
    # Get tile fraction
    if not linear_model:
        da_tile_frac = determine_tile_frac(vic_global_txt).sel(lat=lat, lon=lon)  # [veg, snow]
    else:
        da_tile_frac = xr.DataArray([[1]], coords=[[1], [0]], dims=['veg', 'snow'])
    
    # Initial state perturbation
    init_dir = os.path.join(
                    EnKF_result_basedir, 'temp',
                    'init.{}_{:05d}'.format(init_time.strftime('%Y%m%d'),
                                            init_time.hour*3600+init_time.second))
    list_da = []
    for i in range(N):
        fname = os.path.join(init_dir, 'perturbation.ens{}.nc'.format(i+1))
        da = xr.open_dataset(fname)['STATE_SOIL_MOISTURE'].sel(
                    lat=lat, lon=lon)  # [veg_class, snow_band, nlayer]
        list_da.append(da)
    # Concat all ensemble members together
    da_init_perturbation = xr.concat(list_da, dim='N')
    da_init_perturbation['N'] = range(1, N+1)  # [N, veg_class, snow_band, nlayer]
    
    # State perturbation
    list_da = []
    for i, t in enumerate(times):
        # Load data
        fname = os.path.join(
                    EnKF_result_basedir, 'temp', 'perturbation',
                    'perturbation.{}_{:05d}.nc'.format(
                    t.strftime('%Y%m%d'),
                    t.hour*3600+t.second))
        da = xr.open_dataset(fname).sel(lat=lat, lon=lon)['soil_moisture_perturbation']
            # [N, veg_class, snow_band, nlayer]
        # Put data in array
        list_da.append(da)
    # Concat all time points (together with initial state)
    da_perturbation = xr.concat([da_init_perturbation] + list_da, dim='time')
    da_perturbation['time'] = pd.to_datetime([init_time] + list(times))
        # [time, N, veg_class, snow_band, nlayer]
    # Extract coords
    veg_coord = da_perturbation['veg_class']
    snow_coord = da_perturbation['snow_band']
    nlayer_coord = da_perturbation['nlayer']
    time_coord = da_perturbation['time']
    N_coord = da_perturbation['N']
    # roll nlayer in front of veg and snow
    perturbation = da_perturbation.values
    perturbation = np.rollaxis(perturbation, 4, 2)
    # Put back into a da
    da_perturbation = xr.DataArray(
            perturbation,
            coords=[time_coord, N_coord, nlayer_coord, veg_coord, snow_coord],
            dims=['time', 'N', 'nlayer', 'veg_class', 'snow_band'])
    
    # ### Calculate cell average values ### #
    # Determine the total number of loops
    nloop = len(time_coord) * N * len(nlayer_coord)
    # Convert da_x and da_tile_frac to np.array and straighten lat and lon into nloop
    perturbation = perturbation.reshape(
            [nloop, len(veg_coord), len(snow_coord)])  # [nloop, nveg, nsnow]
    tile_frac = da_tile_frac.values  # [nveg, nsnow]
    # Calculate cell-average value
    perturbation_cellAvg = np.array(list(map(
                lambda i: np.nansum(perturbation[i, :, :] * tile_frac),
                range(nloop))))  # [nloop]
    # Reshape
    perturbation_cellAvg = perturbation_cellAvg.reshape(
            [len(time_coord), N, len(nlayer_coord)])  # [time, N, nlayer]
    # Put in da
    da_perturbation_cellAvg = xr.DataArray(
            perturbation_cellAvg,
            coords=[time_coord, N_coord, nlayer_coord],
            dims=['time', 'N', 'nlayer'])
    
    # --- Diagnostics - update increment --- #
    print('\tDiagnostics - update increments...')
    
    list_da = []
    for i, t in enumerate(times):
        # Load data
        fname = os.path.join(
                    EnKF_result_basedir, 'temp', 'update',
                    'update_increm.{}_{:05d}.nc'.format(
                    t.strftime('%Y%m%d'),
                    t.hour*3600+t.second))
        da = xr.open_dataset(fname).sel(lat=lat, lon=lon)['update_increment']
            # [N, n]
        # Put data in array
        list_da.append(da)
    # Concat all time points (together with initial state)
    da_update_increm = xr.concat(list_da, dim='time')  # [time, N, n]
    da_update_increm['time'] = times
    # Extract coords
    N_coord = da_update_increm['N']
    # Reshape da and reorder dimensions
    update_increm = da_update_increm.values.reshape(
                            [len(times), len(N_coord), len(nlayer_coord),
                             len(veg_coord), len(snow_coord)])
    # Put data back to a da [time, N, nlayer, veg_class, snow_band]
    da_update_increm = xr.DataArray(update_increm,
                                    coords=[times, N_coord, nlayer_coord,
                                            veg_coord, snow_coord],
                                    dims=['time', 'N', 'nlayer', 'veg_class', 'snow_band'])
    
    # ### Calculate cell average values ### #
    # Determine the total number of loops
    nloop = len(times) * N * len(nlayer_coord)
    # Convert da_x and da_tile_frac to np.array and straighten lat and lon into nloop
    update_increm = update_increm.reshape(
            [nloop, len(veg_coord), len(snow_coord)])  # [nloop, nveg, nsnow]
    tile_frac = da_tile_frac.values  # [nveg, nsnow]
    # Calculate cell-average value
    update_increm_cellAvg = np.array(list(map(
                lambda i: np.nansum(update_increm[i, :, :] * tile_frac),
                range(nloop))))  # [nloop]
    # Reshape
    update_increm_cellAvg = update_increm_cellAvg.reshape(
            [len(times), N, len(nlayer_coord)])  # [time, N, nlayer]
    # Put in da
    da_update_increm_cellAvg = xr.DataArray(
            update_increm_cellAvg,
            coords=[times, N_coord, nlayer_coord],
            dims=['time', 'N', 'nlayer'])
    
    # --- Diagnostics - gain K --- #
    print('\tDiagnostics - gain K...')
    list_da = []
    for i, t in enumerate(times):
        # Load data
        fname = os.path.join(
                    EnKF_result_basedir, 'temp', 'update',
                    'K.{}_{:05d}.nc'.format(
                    t.strftime('%Y%m%d'),
                    t.hour*3600+t.second))
        da = xr.open_dataset(fname).sel(lat=lat, lon=lon)['K']
            # [n, m=1]
        # Put data in array
        list_da.append(da)
    # Concat all time points (together with initial state)
    da_K = xr.concat(list_da, dim='time')  # [time, n, m=1]
    da_K['time'] = times
    # Reshape da and reorder dimensions
    K = da_K.values.reshape(
                        [len(times), len(nlayer_coord),
                         len(veg_coord), len(snow_coord)])
    # Put data back to a da [time, veg_class, snow_band, nlayer]
    da_K = xr.DataArray(K,
                        coords=[times, nlayer_coord,
                                veg_coord, snow_coord],
                        dims=['time', 'nlayer', 'veg_class', 'snow_band'])
    
    # ### Calculate cell average values ### #
    # Determine the total number of loops
    nloop = len(times) * len(nlayer_coord)
    # Convert da_x and da_tile_frac to np.array and straighten lat and lon into nloop
    K = K.reshape(
            [nloop, len(veg_coord), len(snow_coord)])  # [nloop, nveg, nsnow]
    tile_frac = da_tile_frac.values  # [nveg, nsnow]
    # Calculate cell-average value
    K_cellAvg = np.array(list(map(
                lambda i: np.nansum(K[i, :, :] * tile_frac),
                range(nloop))))  # [nloop]
    # Reshape
    K_cellAvg = K_cellAvg.reshape(
            [len(times), len(nlayer_coord)])  # [time, nlayer]
    # Put in da
    da_K_cellAvg = xr.DataArray(
            K_cellAvg,
            coords=[times, nlayer_coord],
            dims=['time', 'nlayer'])
    
    # --- Diagnostics - measurement perturbation v --- #
    print('\tDiagnostics - measurement perturbation...')
    
    list_da = []
    for i, t in enumerate(times):
        # Load data
        fname = os.path.join(
                    EnKF_result_basedir, 'temp', 'update',
                    'meas_perturbation.{}_{:05d}.nc'.format(
                    t.strftime('%Y%m%d'),
                    t.hour*3600+t.second))
        da = xr.open_dataset(fname).sel(lat=lat, lon=lon)['meas_perturbation']
            # [N, n]
        # Put data in array
        list_da.append(da)
    # Concat all time points (together with initial state)
    da_v = xr.concat(list_da, dim='time')  # [time, N, m=1]
    da_v['time'] = times
    # Extract coords
    N_coord = da_v['N']
    # Reshape da - [time, N]
    v = da_v.values.reshape([len(times), len(N_coord)])
    # Put data back to a da [time, N]
    da_v = xr.DataArray(v,
                        coords=[times, N_coord],
                        dims=['time', 'N'])


# ========================================================== #
# Plot - innovation
# ========================================================== #
print('Plotting...')
print('\tPlot - innovation...')
fig = plt.figure(figsize=(12, 6))
s_innov_norm.plot(color='g', style='-',
                  label='Innovation (meas - y_est_before_update)\n'
                  'mean={:.2f}, var_norm={:.2f}'.format(
                        s_innov_norm.mean(), s_innov_norm.var()),
                        legend=True)
plt.xlabel('Time')
plt.ylabel('Innovation (-)')
plt.title('Normalized innovation, {}, {}, N={}'.format(lat, lon, N))
fig.savefig(os.path.join(output_dir, '{}_{}.innov.png'.format(lat, lon)),
            format='png')

# Plot innovation autocorrolation (ACF)
fig = plt.figure(figsize=(12, 6))
pd.tools.plotting.autocorrelation_plot(s_innov_norm)
plt.xlabel('Lag (day)')
plt.xlim([0, 300])
plt.title('Innovation ACF, {}, {}, N={}'.format(lat, lon, N))
fig.savefig(os.path.join(output_dir, '{}_{}.innov_acf.png'.format(lat, lon)),
            format='png')

# ========================================================== #
# Plot - precipitation
# ========================================================== #
print('\tPlot - precipitation...')

da_prec_ens = ds_force_ens['PREC']
ts_prec_orig = ds_force_orig['PREC'].to_series()
ts_prec_truth = ds_force_truth['PREC'].to_series()

# Calculate EnKF_mean vs. truth
df_truth_EnKF = pd.concat([ts_prec_truth, da_prec_ens.mean(dim='N').to_series()],
                          axis=1, keys=['truth', 'EnKF_mean']).dropna()
rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_prec_truth, ts_prec_orig], axis=1,
                              keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.prec.html'.format(lat, lon)))

p = figure(title='Precipitation, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Precipitation (mm/step)",
           x_axis_type='datetime', width=1000, height=500)
# plot each ensemble member
for i in range(N):
    ens_name = 'ens{}'.format(i+1)
    if i == 0:
        legend="Ensemble members (perturbed from Newman ens. 100)\n" \
               "mean RMSE = {:.3f} mm".format(rmse_EnKF_mean)
    else:
        legend=False
    ts = da_prec_ens.sel(N=i+1).to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot truth
ts = ts_prec_truth
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Truth (perturbed from Newman ens. 100)", line_width=2)
# plot orig.
ts = ts_prec_orig
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Orig. (Newman ens. 100)\n" \
              "mean RMSE = {:.3f} mm".format(rmse_openloop),
       line_width=2)
# Save
save(p)

# ========================================================== #
# Plot - precipitation, aggregated to daily
# ========================================================== #
print('\tPlot - precipitation, aggregated to daily...')

da_prec_ens = ds_force_ens['PREC'].resample(freq="D", how='sum', dim='time')
ts_prec_orig = ds_force_orig['PREC'].to_series().resample("D", how='sum')
ts_prec_truth = ds_force_truth['PREC'].to_series().resample("D", how='sum')

# Calculate EnKF_mean vs. truth
df_truth_EnKF = pd.concat([ts_prec_truth, da_prec_ens.mean(dim='N').to_series()],
                          axis=1, keys=['truth', 'EnKF_mean']).dropna()
rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
# Calculate open-loop vs. truth
df_truth_openloop = pd.concat([ts_prec_truth, ts_prec_orig], axis=1,
                              keys=['truth', 'openloop']).dropna()
rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.prec_daily.html'.format(lat, lon)))

p = figure(title='Precipitation, daily, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Precipitation (mm/step)",
           x_axis_type='datetime', width=1000, height=500)
# plot each ensemble member
for i in range(N):
    ens_name = 'ens{}'.format(i+1)
    if i == 0:
        legend="Ensemble members (perturbed from Newman ens. 100)\n" \
               "mean RMSE = {:.3f} mm".format(rmse_EnKF_mean)
    else:
        legend=False
    ts = da_prec_ens.sel(N=i+1).to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot truth
ts = ts_prec_truth
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Truth (perturbed from Newman ens. 100)", line_width=2)
# plot orig.
ts = ts_prec_orig
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Orig. (Newman ens. 100)\n" \
              "mean RMSE = {:.3f} mm".format(rmse_openloop),
       line_width=2)
# Save
save(p)

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
da_EnKF = ds_EnKF['OUT_SOIL_MOIST'].sel(nlayer=0,
                                        time=slice(plot_start_time, plot_end_time)) / depth_sm1
ts_EnKF_mean = da_EnKF.mean(dim='N').\
               to_series()
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=0,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
rmse_meas = rmse(df_truth_meas['truth'].values, df_truth_meas['meas'])
# Calculate EnKF_mean vs. truth
df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
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
    da_EnKF.sel(N=i+1).to_series().plot(
                color='blue', style='-', alpha=0.3,
                label='Ensemble members, mean RMSE={:.3f} mm/mm'.format(rmse_EnKF_mean),
                legend=legend)
# plot measurement
ts_meas.plot(style='ro', label='Measurement, RMSE={:.3f} mm/mm'.format(rmse_meas),
             legend=True)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='-',
                 label='Open-loop, RMSE={:.3f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Plot bias correction reference
if bias_correct:
    ts_bc_ref = ds_hist_ref['OUT_SOIL_MOIST'].sel(
        nlayer=0,
        time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
    ts_bc_ref.plot(color='orange', style='-',
        label='Bias correction reference',
        legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Top-layer soil moisture, {}, {}, N={}'.format(lat, lon, N))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.sm1.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm1.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each ensemble member
for i in range(N):
    ens_name = 'ens{}'.format(i+1)
    if i == 0:
        legend="Ensemble members, mean RMSE={:.2f} mm/mm".format(rmse_EnKF_mean)
    else:
        legend=False
    ts = da_EnKF.sel(N=i+1).to_series()
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
# Plot bias correction reference
if bias_correct:
    ts = ds_hist_ref['OUT_SOIL_MOIST'].sel(
        nlayer=0,
        time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm1
    p.line(ts.index, ts.values, color="orange", line_dash="solid",
        legend="Bias correction reference", line_width=2)
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
da_EnKF = ds_EnKF['OUT_SOIL_MOIST'].sel(nlayer=1,
                                        time=slice(plot_start_time, plot_end_time)) / depth_sm2
ts_EnKF_mean = da_EnKF.mean(dim='N').\
               to_series()
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=1,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm2
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
# Calculate EnKF_mean vs. truth
df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
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
    da_EnKF.sel(N=i+1).to_series().plot(
                color='blue', style='-', alpha=0.3,
                label='Ensemble members, mean RMSE={:.5f} mm/mm'.format(rmse_EnKF_mean),
                legend=legend)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.5f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Plot bias correction reference
if bias_correct:
    ts_bc_ref = ds_hist_ref['OUT_SOIL_MOIST'].sel(
        nlayer=1,
        time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm2
    ts_bc_ref.plot(color='orange', style='-',
        label='Bias correction reference',
        legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Middle-layer soil moisture, {}, {}, N={}'.format(lat, lon, N))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.sm2.png'.format(lat, lon)),
            format='png')

# ----- Interactive version ----- #
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm2.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Soil moiture (mm/mm)",
           x_axis_type='datetime', width=1000, height=500)
# plot each ensemble member
for i in range(N):
    ens_name = 'ens{}'.format(i+1)
    if i == 0:
        legend="Ensemble members, mean RMSE={:.5f} mm/mm".format(rmse_EnKF_mean)
    else:
        legend=False
    ts = da_EnKF.sel(N=i+1).to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# plot truth
ts = ts_truth
p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
       legend="Open-loop, RMSE={:.5f} mm/mm".format(rmse_openloop), line_width=2)
# Plot bias correction reference
if bias_correct:
    ts = ds_hist_ref['OUT_SOIL_MOIST'].sel(
        nlayer=1,
        time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm2
    p.line(ts.index, ts.values, color="orange", line_dash="solid",
        legend="Bias correction reference", line_width=2)
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
da_EnKF = ds_EnKF['OUT_SOIL_MOIST'].sel(nlayer=2,
                                        time=slice(plot_start_time, plot_end_time)) / depth_sm3
ts_EnKF_mean = da_EnKF.mean(dim='N').\
               to_series()
ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                lat=lat, lon=lon, nlayer=2,
                time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm3
# Calculate meas vs. truth
df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
# Calculate EnKF_mean vs. truth
df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
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
    da_EnKF.sel(N=i+1).to_series().plot(
                color='blue', style='-', alpha=0.3,
                label='Ensemble members, mean RMSE={:.5f} mm/mm'.format(rmse_EnKF_mean),
                legend=legend)
# plot truth
ts_truth.plot(color='k', style='-', label='Truth', legend=True)
# plot open-loop
ts_openloop.plot(color='m', style='--',
                 label='Open-loop, RMSE={:.5f} mm/mm'.format(rmse_openloop),
                 legend=True)
# Plot bias correction reference
if bias_correct:
    ts_bc_ref = ds_hist_ref['OUT_SOIL_MOIST'].sel(
        nlayer=2,
        time=slice(plot_start_time, plot_end_time)).to_series() / depth_sm3
    ts_bc_ref.plot(color='orange', style='-',
        label='Bias correction reference',
        legend=True)
# Make plot looks better
plt.xlabel('Time')
plt.ylabel('Soil moiture (mm/mm)')
plt.title('Bottom-layer soil moisture, {}, {}, N={}'.format(lat, lon, N))
# Save figure
fig.savefig(os.path.join(output_dir, '{}_{}.sm3.png'.format(lat, lon)),
            format='png')

# ========================================================== #
# Plot - bias correction
# ========================================================== #
if bias_correct:
    print('\tPlot - bias correction...')
    # --- Regular plot --- #
    fig = plt.figure(figsize=(12, 6))
    da_delta_cellAvg.sel(nlayer=0).to_series().plot(
            color='b', style='-',
            label='Layer 1', legend=True)
    da_delta_cellAvg.sel(nlayer=1).to_series().plot(
            color='orange', style='-',
            label='Layer 2', legend=True)
    da_delta_cellAvg.sel(nlayer=2).to_series().plot(
            color='green', style='-',
            label='Layer 3', legend=True)
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('Bias correction delta of soil moistures, {}, {}, N={}'.format(lat, lon, N))
    fig.savefig(os.path.join(output_dir, '{}_{}.bc_delta.sm.png'.format(lat, lon)),
            format='png')
    # --- Interactive plot --- #
    output_file(os.path.join(output_dir, '{}_{}.bc_delta.sm.html'.format(lat, lon)))
    p = figure(title='Bias correction delta of soil moistures, {}, {}, N={}'.format(lat, lon, N),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
    # sm1
    ts = da_delta_cellAvg.sel(nlayer=0).to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid",
           legend="Layer 1", line_width=2)
    # sm2
    ts = da_delta_cellAvg.sel(nlayer=1).to_series()
    p.line(ts.index, ts.values, color="orange", line_dash="solid",
           legend="Layer 2", line_width=2)
    # sm3
    ts = da_delta_cellAvg.sel(nlayer=2).to_series()
    p.line(ts.index, ts.values, color="green", line_dash="solid",
           legend="Layer 3", line_width=2)
    # save
    save(p)

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
    da_EnKF = ds_EnKF['OUT_RUNOFF'].sel(time=slice(plot_start_time, plot_end_time))
    ts_EnKF_mean = da_EnKF.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_RUNOFF'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate EnKF_mean vs. truth
    df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
    rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.runoff.html'.format(lat, lon)))
    
    p = figure(title='Surface runoff, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="Runoff (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Ensemble members, mean RMSE={:.3f} mm".format(rmse_EnKF_mean)
        else:
            legend=False
        ts = da_EnKF.sel(N=i+1).to_series()
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
if not linear_model:
    print('\tPlot - surface runoff, aggregated to daily...')
    # --- RMSE --- #
    # extract time series
    ts_truth = ds_truth['OUT_RUNOFF'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series().\
               resample("D", how='sum')
    da_EnKF = ds_EnKF['OUT_RUNOFF'].sel(time=slice(plot_start_time, plot_end_time)).\
              resample(freq="D", how='sum', dim='time')
    ts_EnKF_mean = da_EnKF.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_RUNOFF'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series().\
                  resample("D", how='sum')
    # Calculate EnKF_mean vs. truth
    df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
    rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.runoff_daily.html'.format(lat, lon)))
    
    p = figure(title='Surface runoff, daily, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="Runoff (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Ensemble members, mean RMSE={:.3f} mm".format(rmse_EnKF_mean)
        else:
            legend=False
        ts = da_EnKF.sel(N=i+1).to_series()
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
    da_EnKF = ds_EnKF['OUT_BASEFLOW'].sel(time=slice(plot_start_time, plot_end_time))
    ts_EnKF_mean = da_EnKF.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_BASEFLOW'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate EnKF_mean vs. truth
    df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
    rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
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
        da_EnKF.sel(N=i+1).to_series().plot(
                    color='blue', style='-', alpha=0.3,
                    label='Ensemble members, mean RMSE={:.3f} mm'.format(rmse_EnKF_mean),
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
    fig.savefig(os.path.join(output_dir, '{}_{}.baseflow.png'.format(lat, lon)),
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
    da_EnKF = (ds_EnKF['OUT_RUNOFF'] + ds_EnKF['OUT_BASEFLOW']).sel(
                    time=slice(plot_start_time, plot_end_time))
    ts_EnKF_mean = da_EnKF.mean(dim='N').\
                   to_series()
    ts_openloop = (ds_openloop['OUT_RUNOFF'] + ds_openloop['OUT_BASEFLOW']).sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate EnKF_mean vs. truth
    df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
    rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.total_runoff.html'.format(lat, lon)))
    
    p = figure(title='Total runoff, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="Total runoff (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Ensemble members, mean RMSE={:.3f} mm".format(rmse_EnKF_mean)
        else:
            legend=False
        ts = da_EnKF.sel(N=i+1).to_series()
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
    da_EnKF = ds_EnKF['OUT_SWE'].sel(time=slice(plot_start_time, plot_end_time))
    ts_EnKF_mean = da_EnKF.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_SWE'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate EnKF_mean vs. truth
    df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
    rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.swe.html'.format(lat, lon)))
    
    p = figure(title='Surface SWE, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="SWE (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Ensemble members, mean RMSE={:.3f} mm".format(rmse_EnKF_mean)
        else:
            legend=False
        ts = da_EnKF.sel(N=i+1).to_series()
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
    da_EnKF = ds_EnKF['OUT_EVAP'].sel(time=slice(plot_start_time, plot_end_time))
    ts_EnKF_mean = da_EnKF.mean(dim='N').\
                   to_series()
    ts_openloop = ds_openloop['OUT_EVAP'].sel(
                    lat=lat, lon=lon,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    # Calculate EnKF_mean vs. truth
    df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
    rmse_EnKF_mean = rmse(df_truth_EnKF['truth'], df_truth_EnKF['EnKF_mean'])
    # Calculate open-loop vs. truth
    df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
    rmse_openloop = rmse(df_truth_openloop['truth'], df_truth_openloop['openloop'])
    
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.evap.html'.format(lat, lon)))
    
    p = figure(title='ET, {}, {}, N={}'.format(lat, lon, N),
               x_axis_label="Time", y_axis_label="ET (mm)",
               x_axis_type='datetime', width=1000, height=500)
    # plot each ensemble member
    for i in range(N):
        ens_name = 'ens{}'.format(i+1)
        if i == 0:
            legend="Ensemble members, mean RMSE={:.3f} mm".format(rmse_EnKF_mean)
        else:
            legend=False
        ts = da_EnKF.sel(N=i+1).to_series()
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
# Plot - diagnostics
# ========================================================== #
if debug:
    print('\tPlot - diagnostics...')
    # --- Perturbation --- #
    fig = plt.figure(figsize=(12, 5))
    s1 = da_perturbation_cellAvg.sel(N=ens, nlayer=0).to_series()
    s1.plot(color='b',
            label='Layer 1, mean={:.2f}, var={:.2f}'.format(s1.mean(), s1.var()),
            legend=True)
    s2 = da_perturbation_cellAvg.sel(N=ens, nlayer=1).to_series()
    s2.plot(color='g',
            label='Layer 2, mean={:.2f}, var={:.2f}'.format(s2.mean(), s2.var()),
            legend=True)
    s3 = da_perturbation_cellAvg.sel(N=ens, nlayer=2).to_series()
    s3.plot(color='r',
            label='Layer 3, mean={:.2f}, var={:.2f}'.format(s3.mean(), s3.var()),
            legend=True)
    # Make plot looks better
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('Soil moisture perturbation amount, {}, {}, ens. {}'.format(lat, lon, ens))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.perturbation.ens{}.png'.format(lat, lon, ens)),
                format='png')
    
    # --- Gain K --- #
    fig = plt.figure(figsize=(12, 5))
    s_K_cellAvg_1 = da_K_cellAvg.sel(nlayer=0).to_series()
    s_K_cellAvg_2 = da_K_cellAvg.sel(nlayer=1).to_series()
    s_K_cellAvg_3 = da_K_cellAvg.sel(nlayer=2).to_series()
    
    s_K_cellAvg_1.plot(color='b', label='Layer 1', legend=True)
    s_K_cellAvg_2.plot(color='g', label='Layer 2', legend=True)
    s_K_cellAvg_3.plot(color='r', label='Layer 3', legend=True)
    # Make plot looks better
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Gain K')
    plt.title('Gain K, {}, {}'.format(lat, lon))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.K.png'.format(lat, lon)),
                format='png')
    # ----- Interactive version ----- #
    # Create figure
    output_file(os.path.join(output_dir, '{}_{}.debug.K.html'.format(lat, lon)))

    p = figure(title='Gain K, {}, {}'.format(lat, lon),
               x_axis_label="Time", y_axis_label="Gain K",
               x_axis_type='datetime', width=1000, height=500)
    # plot
    p.line(s_K_cellAvg_1.index, s_K_cellAvg_1.values, color="blue",
           line_dash="solid", legend="Layer 1", line_width=2)
    p.line(s_K_cellAvg_2.index, s_K_cellAvg_2.values, color="green",
           line_dash="solid", legend="Layer 2", line_width=2)
    p.line(s_K_cellAvg_3.index, s_K_cellAvg_3.values, color="red",
           line_dash="solid", legend="Layer 3", line_width=2)
    # Save
    save(p)
    
    # --- Scatter plot --- #
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(131)
    plt.scatter(s_K_cellAvg_1, s_K_cellAvg_2)
    plt.xlabel('Gain K, layer 1')
    plt.ylabel('Gain K, layer 2')
    plt.subplot(132)
    plt.scatter(s_K_cellAvg_1, s_K_cellAvg_3)
    plt.xlabel('Gain K, layer 1')
    plt.ylabel('Gain K, layer 3')
    plt.title('Gain K vertical relation, {}, {}'.format(lat, lon),
              fontsize=20)
    plt.subplot(133)
    plt.scatter(s_K_cellAvg_2, s_K_cellAvg_3)
    plt.xlabel('Gain K, layer 2')
    plt.ylabel('Gain K, layer 3')
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.K_vertical_relation.png'.format(lat, lon)),
                format='png')
    
    # --- Sum of update increments of layer 1 and layer 2 --- #
    s_update1 = da_update_increm_cellAvg.sel(N=ens, nlayer=0).to_series()
    s_update2 = da_update_increm_cellAvg.sel(N=ens, nlayer=1).to_series()
    
    fig = plt.figure(figsize=(12, 5))
    (s_update1 + s_update2).plot(color='k')
    # Make plot looks better
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('(Layer 1 + layer 2) update amount, {}, {}, ens. {}'.format(lat, lon, ens))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.update_increm_sm12.ens{}.png'.format(lat, lon, ens)),
                format='png')
    
    # ========================================================== #
    # Plot - diagnostics
    # ========================================================== #
    # --- Perturbation --- #
    ens = 1
    
    fig = plt.figure(figsize=(12, 5))
    s1 = da_update_increm_cellAvg.sel(N=ens, nlayer=0).to_series()
    s1.plot(color='b',
            label='Layer 1, mean={:.2f}, var={:.2f}'.format(s1.mean(), s1.var()),
            legend=True)
    s2 = da_update_increm_cellAvg.sel(N=ens, nlayer=1).to_series()
    s2.plot(color='g',
            label='Layer 2, mean={:.2f}, var={:.2f}'.format(s2.mean(), s2.var()),
            legend=True)
    s3 = da_update_increm_cellAvg.sel(N=ens, nlayer=2).to_series()
    s3.plot(color='r',
            label='Layer 3, mean={:.2f}, var={:.2f}'.format(s3.mean(), s3.var()),
            legend=True)
    # Make plot looks better
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('Soil moisture Kalman update amount, {}, {}, ens. {}'.format(lat, lon, ens))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.update_increm.ens{}.png'.format(lat, lon, ens)),
                format='png')
    
    # --- Compare perturb & update, sm1 --- #
    s_perturbation = da_perturbation_cellAvg.sel(N=ens, nlayer=0).to_series()
    s_update = da_update_increm_cellAvg.sel(N=ens, nlayer=0).to_series()
    
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes()
    s_perturbation.plot(color='b',
                        label='Perturbation, sum={:.2f} mm'.format(s_perturbation.sum()))
    s_update.plot(color='g',
                  label='Update increm., sum={:.2f} mm'.format(s_update.sum()))
    # Calculate mean diff of openloop and truth
    ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                    lat=lat, lon=lon, nlayer=0,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(
                    lat=lat, lon=lon, nlayer=0,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    diff_openloop_truth = (ts_openloop - ts_truth).mean()
    # Make plot better
    # Make plot looks better
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('sm1, ens. {}\n{}, {}, avg.(openloop - truth) = {:.1f} mm'.format(
                    ens, lat, lon, diff_openloop_truth))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.cmp_pert_update.sm1.ens{}.png'.format(lat, lon, ens)),
                format='png')
    
    # --- Compare perturb & update, sm2 --- #
    s_perturbation = da_perturbation_cellAvg.sel(N=ens, nlayer=1).to_series()
    s_update = da_update_increm_cellAvg.sel(N=ens, nlayer=1).to_series()
    
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes()
    s_perturbation.plot(color='b',
                        label='Perturbation, sum={:.2f} mm'.format(s_perturbation.sum()))
    s_update.plot(color='g',
                  label='Update increm., sum={:.2f} mm'.format(s_update.sum()))
    # Calculate mean diff of openloop and truth
    ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                    lat=lat, lon=lon, nlayer=1,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(
                    lat=lat, lon=lon, nlayer=1,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    diff_openloop_truth = (ts_openloop - ts_truth).mean()
    # Make plot better
    # Make plot looks better
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('sm2, ens. {}\n{}, {}, avg.(openloop - truth) = {:.1f} mm'.format(
                    ens, lat, lon, diff_openloop_truth))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.cmp_pert_update.sm2.ens{}.png'.format(lat, lon, ens)),
                format='png')
    
    # --- Compare perturb & update, sm3 --- #
    s_perturbation = da_perturbation_cellAvg.sel(N=ens, nlayer=2).to_series()
    s_update = da_update_increm_cellAvg.sel(N=ens, nlayer=2).to_series()
    
    fig = plt.figure(figsize=(12, 5))
    ax = plt.axes()
    s_perturbation.plot(color='b',
                        label='Perturbation, sum={:.2f} mm'.format(s_perturbation.sum()))
    s_update.plot(color='g',
                  label='Update increm., sum={:.2f} mm'.format(s_update.sum()))
    # Calculate mean diff of openloop and truth
    ts_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(
                    lat=lat, lon=lon, nlayer=2,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    ts_truth = ds_truth['OUT_SOIL_MOIST'].sel(
                    lat=lat, lon=lon, nlayer=2,
                    time=slice(plot_start_time, plot_end_time)).to_series()
    diff_openloop_truth = (ts_openloop - ts_truth).mean()
    # Make plot better
    # Make plot looks better
    plt.legend(loc='upper right')
    plt.xlabel('Time')
    plt.ylabel('Soil moiture (mm)')
    plt.title('sm3, ens. {}\n{}, {}, avg.(openloop - truth) = {:.1f} mm'.format(
                    ens, lat, lon, diff_openloop_truth))
    # Save figure
    fig.savefig(os.path.join(output_dir,
                             '{}_{}.debug.cmp_pert_update.sm3.ens{}.png'.format(lat, lon, ens)),
                format='png')




