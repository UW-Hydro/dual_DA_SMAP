
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

from tonic.io import read_configobj
import timeit

from analysis_utils import (
    rmse, find_global_param_value, determine_tile_frac, get_soil_depth,
    load_nc_file, setup_output_dirs, calc_sm_runoff_corrcoef,
    to_netcdf_state_file_compress)


# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.args[1])


# ========================================================== #
# Parameter setting
# ========================================================== #
# --- Input directory and files --- #
# gen_synthetic results
gen_synth_basedir = cfg['SYNTHETIC']['gen_synth_basedir']
truth_nc_filename = cfg['SYNTHETIC']['truth_nc_filename']
synth_meas_nc_filename = cfg['SYNTHETIC']['synth_meas_nc_filename']

# openloop
openloop_basedir = cfg['SYNTHETIC']['openloop_basedir']

# Time period
start_time = pd.to_datetime(cfg['SYNTHETIC']['start_time'])
end_time = pd.to_datetime(cfg['SYNTHETIC']['end_time'])

# VIC global file template (for extracting param file and snow_band)
vic_global_txt = cfg['SYNTHETIC']['vic_global_txt']

# Forcings (for all basepaths, 'YYYY.nc' will be appended)
orig_force_basepath = cfg['SYNTHETIC']['orig_force_basepath']
truth_force_basepath = cfg['SYNTHETIC']['truth_force_basepath']
# ens_force_basedir/ens_<i>/force.<YYYY>.nc, where <i> = 1, 2, ..., N
ens_force_basedir = cfg['SYNTHETIC']['ens_force_basedir']

# VIC parameter netCDF file
vic_param_nc = cfg['SYNTHETIC']['vic_param_nc']

# Domain netCDF file
domain_nc = cfg['SYNTHETIC']['domain_nc']

# --- Measurement times --- #
meas_times = pd.date_range(
    cfg['SYNTHETIC']['meas_start_time'],
    cfg['SYNTHETIC']['meas_end_time'],
    freq=cfg['SYNTHETIC']['freq'])

# --- Plot time period --- #
plot_start_time = pd.to_datetime(cfg['SYNTHETIC']['plot_start_time'])
plot_end_time = pd.to_datetime(cfg['SYNTHETIC']['plot_end_time'])
plot_start_year = plot_start_time.year
plot_end_year = plot_end_time.year

# --- Output --- #
output_rootdir = cfg['OUTPUT']['output_dir']


# ========================================================== #
# Setup output data dir
# ========================================================== #
output_data_dir = setup_output_dirs(
        output_rootdir,
        mkdirs=['data'])['data']

output_maps_dir = setup_output_dirs(
        output_rootdir,
        mkdirs=['maps'])['maps']


# ========================================================== #
# Load data
# ========================================================== #
print('Loading data...')

# --- Domain --- #
da_domain = xr.open_dataset(domain_nc)['mask']

# --- Tile fraction --- #
da_tile_frac = determine_tile_frac(vic_global_txt)  # [veg_class, snow_band, lat, lon]

# --- Openloop --- #
print('\tOpenloop history...')
openloop_hist_nc = os.path.join(
    openloop_basedir,
    'history',
    'history.openloop.{}-{:05d}.nc'.format(
        start_time.strftime('%Y-%m-%d'),
        start_time.hour*3600+start_time.second))
ds_openloop_hist = xr.open_dataset(openloop_hist_nc)
print('\tOpenloop states...')
openloop_state_nc = os.path.join(
    openloop_basedir,
    'states',
    'openloop_state_cellAvg.{}_{}.nc'.format(
        meas_times[0].strftime('%Y%m%d'),
        meas_times[-1].strftime('%Y%m%d')))
da_openloop_states = xr.open_dataset(openloop_state_nc)['SOIL_MOISTURE']

# --- Truth --- #
print('\tTruth history...')
ds_truth_hist = xr.open_dataset(os.path.join(
        gen_synth_basedir, 'truth',
        'history', truth_nc_filename))
print('\tTruth states...')
truth_state_nc = os.path.join(
    gen_synth_basedir,
    'truth',
    'states',
    'truth_state_cellAvg.{}_{}.nc'.format(
        meas_times[0].strftime('%Y%m%d'),
        meas_times[-1].strftime('%Y%m%d')))
da_truth_states = xr.open_dataset(truth_state_nc)['SOIL_MOISTURE']

# --- Perfect-all-state, orig. forcing --- #
print('\ttruthStates_origP...')
ds_truthStateOrigP_hist = xr.open_dataset(os.path.join(
    gen_synth_basedir,
    'test.truth_states_orig_forcing',
    'history',
    'history.concat.{}_{}.nc'.format(
        start_time.strftime('%Y%m%d'),
        end_time.strftime('%Y%m%d'))))


# ======================================================== #
# Extract shared coordinates
# ======================================================== #
lat_coord = da_domain['lat']
lon_coord = da_domain['lon']


# ======================================================== #
# Extract soil layer depths
# ======================================================== #
da_soil_depth = get_soil_depth(vic_param_nc)  # [nlayer, lat, lon]
depth_sm1 = da_soil_depth.sel(nlayer=0)  # [lat, lon]
depth_sm2 = da_soil_depth.sel(nlayer=1)  # [lat, lon]
depth_sm3 = da_soil_depth.sel(nlayer=2)  # [lat, lon]


# ======================================================== #
# Calculate RMSE(SM_openloop)
# ======================================================== #
print('Calculating RMSE(SM_openloop)')
# === sm1 === #
print('\tsm1')
out_nc = os.path.join(output_data_dir, 'rmse_openloop_sm1.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = da_truth_states.sel(nlayer=0) / depth_sm1
    da_openloop = da_openloop_states.sel(nlayer=0) / depth_sm1
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    openloop = da_openloop.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_openloop = np.array(list(map(
                 lambda j: rmse(truth[:, j], openloop[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_openloop = rmse_openloop.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_openloop_sm1 = xr.DataArray(
        rmse_openloop, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)  # [mm/mm]
    # Save RMSE to netCDF file
    ds_rmse_openloop_sm1 = xr.Dataset(
        {'rmse_openloop_sm1': da_rmse_openloop_sm1})
    ds_rmse_openloop_sm1.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:  # if RMSE is already calculated
    da_rmse_openloop_sm1 = xr.open_dataset(out_nc)['rmse_openloop_sm1']
    
# === sm2 === #
print('\tsm2')
out_nc = os.path.join(output_data_dir, 'rmse_openloop_sm2.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = da_truth_states.sel(nlayer=1) / depth_sm2
    da_openloop = da_openloop_states.sel(nlayer=1) / depth_sm2
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    openloop = da_openloop.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_openloop = np.array(list(map(
                 lambda j: rmse(truth[:, j], openloop[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_openloop = rmse_openloop.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_openloop_sm2 = xr.DataArray(
        rmse_openloop, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)  # [mm/mm]
    # Save RMSE to netCDF file
    ds_rmse_openloop_sm2 = xr.Dataset(
        {'rmse_openloop_sm2': da_rmse_openloop_sm2})
    ds_rmse_openloop_sm2.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:  # if RMSE is already calculated
    da_rmse_openloop_sm2 = xr.open_dataset(out_nc)['rmse_openloop_sm2']
    
# === sm3 === #
print('\tsm3')
out_nc = os.path.join(output_data_dir, 'rmse_openloop_sm3.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = da_truth_states.sel(nlayer=2) / depth_sm3
    da_openloop = da_openloop_states.sel(nlayer=2) / depth_sm3
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    openloop = da_openloop.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_openloop = np.array(list(map(
                 lambda j: rmse(truth[:, j], openloop[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_openloop = rmse_openloop.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_openloop_sm3 = xr.DataArray(
        rmse_openloop, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)  # [mm/mm]
    # Save RMSE to netCDF file
    ds_rmse_openloop_sm3 = xr.Dataset(
        {'rmse_openloop_sm3': da_rmse_openloop_sm3})
    ds_rmse_openloop_sm3.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:  # if RMSE is already calculated
    da_rmse_openloop_sm3 = xr.open_dataset(out_nc)['rmse_openloop_sm3']


# ======================================================== #
# Calculate PBIAS(SM_openloop)
# ======================================================== #
print('Calculating PBIAS(SM_openloop)')
# === sm1 === #
print('\tsm1')
out_nc = os.path.join(output_data_dir, 'pbias_openloop_sm1.nc')
if not os.path.isfile(out_nc):  # if PBIAS is not already calculated
    # --- Extract variables --- #
    da_truth = da_truth_states.sel(nlayer=0) / depth_sm1
    da_openloop = da_openloop_states.sel(nlayer=0) / depth_sm1
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_openloop_mean = da_openloop.mean(dim='time')
    da_pbias_openloop_sm1 = (da_openloop_mean - da_truth_mean) / da_truth_mean * 100
    # Save PBIAS to netCDF file
    ds_pbias_openloop_sm1 = xr.Dataset(
         {'pbias_openloop_sm1': da_pbias_openloop_sm1})
    ds_pbias_openloop_sm1.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_openloop_sm1 = xr.open_dataset(out_nc)['pbias_openloop_sm1']

# === sm2 === #
print('\tsm2')
out_nc = os.path.join(output_data_dir, 'pbias_openloop_sm2.nc')
if not os.path.isfile(out_nc):  # if PBIAS is not already calculated
    # --- Extract variables --- #
    da_truth = da_truth_states.sel(nlayer=1) / depth_sm2
    da_openloop = da_openloop_states.sel(nlayer=1) / depth_sm2
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_openloop_mean = da_openloop.mean(dim='time')
    da_pbias_openloop_sm2 = (da_openloop_mean - da_truth_mean) / da_truth_mean * 100
    # Save PBIAS to netCDF file
    ds_pbias_openloop_sm2 = xr.Dataset(
         {'pbias_openloop_sm2': da_pbias_openloop_sm2})
    ds_pbias_openloop_sm2.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_openloop_sm2 = xr.open_dataset(out_nc)['pbias_openloop_sm2']
    
# === sm3 === #
print('\tsm3')
out_nc = os.path.join(output_data_dir, 'pbias_openloop_sm3.nc')
if not os.path.isfile(out_nc):  # if PBIAS is not already calculated
    # --- Extract variables --- #
    da_truth = da_truth_states.sel(nlayer=2) / depth_sm3
    da_openloop = da_openloop_states.sel(nlayer=2) / depth_sm3
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_openloop_mean = da_openloop.mean(dim='time')
    da_pbias_openloop_sm3 = (da_openloop_mean - da_truth_mean) / da_truth_mean * 100
    # Save PBIAS to netCDF file
    ds_pbias_openloop_sm3 = xr.Dataset(
         {'pbias_openloop_sm3': da_pbias_openloop_sm3})
    ds_pbias_openloop_sm3.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_openloop_sm3 = xr.open_dataset(out_nc)['pbias_openloop_sm3']


# ======================================================== #
# Calculate RMSE(runoff_openloop) - daily
# ======================================================== #
print('Calculating RMSE(runoff_openloop) - daily')
# === Surface runoff === #
print('\tsurface runoff')
out_nc = os.path.join(output_data_dir, 'rmse_openloop_dailyRunoff.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    da_openloop = ds_openloop_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    openloop = da_openloop.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_openloop = np.array(list(map(
                lambda j: rmse(truth[:, j], openloop[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_openloop = rmse_openloop.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_openloop_dailyRunoff = xr.DataArray(
        rmse_openloop, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)
    # Save RMSE to netCDF file
    ds_rmse_openloop_dailyRunoff = xr.Dataset(
        {'rmse_openloop_daily_runoff': da_rmse_openloop_dailyRunoff})
    ds_rmse_openloop_dailyRunoff.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_rmse_openloop_dailyRunoff = xr.open_dataset(out_nc)\
                                   ['rmse_openloop_daily_runoff']
    
# === Baseflow === #
print('\tbaseflow')
out_nc = os.path.join(output_data_dir, 'rmse_openloop_dailyBaseflow.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    da_openloop = ds_openloop_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    openloop = da_openloop.values.reshape(
        [len(da_openloop['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_openloop = np.array(list(map(
                lambda j: rmse(truth[:, j], openloop[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_openloop = rmse_openloop.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_openloop_dailyBaseflow = xr.DataArray(
        rmse_openloop, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)
    # Save RMSE to netCDF file
    ds_rmse_openloop_dailyBaseflow = xr.Dataset(
        {'rmse_openloop_daily_baseflow': da_rmse_openloop_dailyBaseflow})
    ds_rmse_openloop_dailyBaseflow.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_rmse_openloop_dailyBaseflow = xr.open_dataset(out_nc)\
                                     ['rmse_openloop_daily_baseflow']


# ======================================================== #
# Calculate PBIAS(runoff_openloop) - daily
# ======================================================== #
print('Calculating PBIAS(runoff_openloop) - daily')
# === Surface runoff === #
print('\tsurface runoff')
out_nc = os.path.join(output_data_dir, 'pbias_openloop_dailyRunoff.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    da_openloop = ds_openloop_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_openloop_mean = da_openloop.mean(dim='time')
    da_pbias_openloop_dailyRunoff = \
        (da_openloop_mean - da_truth_mean) / da_truth_mean * 100
    # --- Save PBIAS to netCDF file --- #
    ds_pbias_openloop_dailyRunoff = xr.Dataset(
        {'pbias_openloop_daily_runoff': da_pbias_openloop_dailyRunoff})
    ds_pbias_openloop_dailyRunoff.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_openloop_dailyRunoff = xr.open_dataset(out_nc)\
                                   ['pbias_openloop_daily_runoff']

# === Baseflow === #
print('\tbaseflow')
out_nc = os.path.join(output_data_dir, 'pbias_openloop_dailyBaseflow.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    da_openloop = ds_openloop_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_openloop_mean = da_openloop.mean(dim='time')
    da_pbias_openloop_dailyBaseflow = \
        (da_openloop_mean - da_truth_mean) / da_truth_mean * 100
    # --- Save PBIAS to netCDF file --- #
    ds_pbias_openloop_dailyBaseflow = xr.Dataset(
        {'pbias_openloop_daily_baseflow': da_pbias_openloop_dailyBaseflow})
    ds_pbias_openloop_dailyBaseflow.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_openloop_dailyBaseflow = xr.open_dataset(out_nc)\
                                   ['pbias_openloop_daily_baseflow']


# ======================================================== #
# Calculate PIMPROVE(runoff_truthState_origP, RMSE) - daily
# ======================================================== #
print('Calculating PIMPROVE(runoff_truthState_origP, RMSE) - daily')

# === Surface runoff === #
print('\tsurface runoff')
# --- Calculate RMSE of truthState_origP --- #
out_nc = os.path.join(output_data_dir, 'rmse_truthStateOrigP_dailyRunoff.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    da_truthStateOrigP = ds_truthStateOrigP_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_truthStateOrigP['time']), nloop])  # [time, nloop]
    truthStateOrigP = da_truthStateOrigP.values.reshape(
        [len(da_truthStateOrigP['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_truthStateOrigP = np.array(list(map(
                lambda j: rmse(truth[:, j], truthStateOrigP[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_truthStateOrigP = rmse_truthStateOrigP.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_truthStateOrigP_dailyRunoff = xr.DataArray(
        rmse_truthStateOrigP, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)
    # Save RMSE to netCDF file
    ds_rmse_truthStateOrigP_dailyRunoff = xr.Dataset(
        {'rmse_truthStateOrigP_daily_runoff': da_rmse_truthStateOrigP_dailyRunoff})
    ds_rmse_truthStateOrigP_dailyRunoff.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_rmse_truthStateOrigP_dailyRunoff = xr.open_dataset(out_nc)\
                                          ['rmse_truthStateOrigP_daily_runoff']
# --- Calculate PIMPROVE --- #
pimprov_truthStateOrigP_dailyRunoff_rmse = \
    (1 - da_rmse_truthStateOrigP_dailyRunoff / da_rmse_openloop_dailyRunoff) * 100
    
# === Baseflow === #
print('\tbaseflow')
# --- Calculate RMSE of truthState_origP --- #
out_nc = os.path.join(output_data_dir, 'rmse_truthStateOrigP_dailyBaseflow.nc')
if not os.path.isfile(out_nc):  # if RMSE is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    da_truthStateOrigP = ds_truthStateOrigP_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate RMSE --- #
    # Determine the total number of loops
    nloop = len(lat_coord) * len(lon_coord)
    # Reshape variables
    truth = da_truth.values.reshape(
        [len(da_truthStateOrigP['time']), nloop])  # [time, nloop]
    truthStateOrigP = da_truthStateOrigP.values.reshape(
        [len(da_truthStateOrigP['time']), nloop])  # [time, nloop]
    # Calculate RMSE for all grid cells
    rmse_truthStateOrigP = np.array(list(map(
                lambda j: rmse(truth[:, j], truthStateOrigP[:, j]),
                range(nloop))))  # [nloop]
    # Reshape RMSE's
    rmse_truthStateOrigP = rmse_truthStateOrigP.reshape(
        [len(lat_coord), len(lon_coord)])  # [lat, lon]
    # Put results into da's
    da_rmse_truthStateOrigP_dailyBaseflow = xr.DataArray(
        rmse_truthStateOrigP, coords=[lat_coord, lon_coord],
        dims=['lat', 'lon']).where(da_domain==1)
    # Save RMSE to netCDF file
    ds_rmse_truthStateOrigP_dailyBaseflow = xr.Dataset(
        {'rmse_truthStateOrigP_daily_baseflow': da_rmse_truthStateOrigP_dailyBaseflow})
    ds_rmse_truthStateOrigP_dailyBaseflow.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_rmse_truthStateOrigP_dailyBaseflow = xr.open_dataset(out_nc)\
                                            ['rmse_truthStateOrigP_daily_baseflow']
# --- Calculate PIMPROVE --- #
pimprov_truthStateOrigP_dailyBaseflow_rmse = \
    (1 - da_rmse_truthStateOrigP_dailyBaseflow / da_rmse_openloop_dailyBaseflow) * 100


# ======================================================== #
# Calculate PIMPROVE(runoff_truthState_origP, PBIAS) - daily
# ======================================================== #
print('Calculating PIMPROVE(runoff_truthState_origP, PBIAS) - daily')

# === Surface runoff === #
print('\tsurface runoff')
# --- Calculate PBIAS of truthState_origP --- #
out_nc = os.path.join(output_data_dir, 'pbias_truthStateOrigP_dailyRunoff.nc')
if not os.path.isfile(out_nc):  # if PBIAS is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    da_truthStateOrigP = ds_truthStateOrigP_hist['OUT_RUNOFF'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_truthStateOrigP_mean = da_truthStateOrigP.mean(dim='time')
    da_pbias_truthStateOrigP_dailyRunoff = \
        (da_truthStateOrigP_mean - da_truth_mean) / da_truth_mean * 100
    # --- Save PBIAS to netCDF file --- #
    ds_pbias_truthStateOrigP_dailyRunoff = xr.Dataset(
        {'pbias_truthStateOrigP_daily_runoff': da_pbias_truthStateOrigP_dailyRunoff})
    ds_pbias_truthStateOrigP_dailyRunoff.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_truthStateOrigP_dailyRunoff = xr.open_dataset(out_nc)\
                                          ['pbias_truthStateOrigP_daily_runoff']
# --- Calculate PIMPROVE --- #
pimprov_truthStateOrigP_dailyRunoff_pbias = \
    (1 - abs(da_pbias_truthStateOrigP_dailyRunoff / \
    da_pbias_openloop_dailyRunoff)) * 100
    
# === Baseflow === #
print('\tbaseflow')
# --- Calculate PBIAS of truthState_origP --- #
out_nc = os.path.join(output_data_dir, 'pbias_truthStateOrigP_dailyBaseflow.nc')
if not os.path.isfile(out_nc):  # if PBIAS is not already calculated
    # --- Extract variables --- #
    da_truth = ds_truth_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    da_truthStateOrigP = ds_truthStateOrigP_hist['OUT_BASEFLOW'].resample(
        '1D', dim='time', how='sum')
    # --- Calculate PBIAS --- #
    da_truth_mean = da_truth.mean(dim='time')
    da_truthStateOrigP_mean = da_truthStateOrigP.mean(dim='time')
    da_pbias_truthStateOrigP_dailyBaseflow = \
        (da_truthStateOrigP_mean - da_truth_mean) / da_truth_mean * 100
    # --- Save PBIAS to netCDF file --- #
    ds_pbias_truthStateOrigP_dailyBaseflow = xr.Dataset(
        {'pbias_truthStateOrigP_daily_baseflow': da_pbias_truthStateOrigP_dailyBaseflow})
    ds_pbias_truthStateOrigP_dailyBaseflow.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
else:
    da_pbias_truthStateOrigP_dailyBaseflow = \
        xr.open_dataset(out_nc)['pbias_truthStateOrigP_daily_baseflow']
# --- Calculate PIMPROVE --- #
pimprov_truthStateOrigP_dailyBaseflow_pbias = \
    (1 - abs(da_pbias_truthStateOrigP_dailyBaseflow / \
    da_pbias_openloop_dailyBaseflow)) * 100


# ======================================================== #
# Plot maps
# ======================================================== #
# --- RMSE(sm1_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop_sm1.where(da_domain==1).plot(
    add_colorbar=False, cmap='cool',
    vmin=0, vmax=0.07)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('RMSE of sm1 state, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir, 'rmse.openloop.sm1.png'), format='png')

# --- RMSE(sm2_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop_sm2.where(da_domain==1).plot(
    add_colorbar=False, cmap='cool',
    vmin=0, vmax=0.1)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('RMSE of sm2 state, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir, 'rmse.openloop.sm2.png'), format='png')

# --- RMSE(sm3_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop_sm3.where(da_domain==1).plot(
    add_colorbar=False, cmap='cool',
    vmin=0, vmax=0.1)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/mm)', fontsize=20)
plt.title('RMSE of sm3 state, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir, 'rmse.openloop.sm3.png'), format='png')

# --- RMSE(runoff_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop_dailyRunoff.where(da_domain==1).plot(
    add_colorbar=False, cmap='cool',
    vmin=0, vmax=3.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/day)', fontsize=20)
plt.title('RMSE of daily surface runoff, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'rmse.openloop.dailyRunoff.png'), format='png')

# --- RMSE(baseflow_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_openloop_dailyBaseflow.where(da_domain==1).plot(
    add_colorbar=False, cmap='cool',
    vmin=0, vmax=0.5)
cbar = plt.colorbar(cs, extend='max').set_label('RMSE (mm/day)', fontsize=20)
plt.title('RMSE of daily baseflow, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'rmse.openloop.dailyBaseflow.png'), format='png')

# --- PIMPROV(runoff_truthStateOrigP, RMSE) --- #
fig = plt.figure(figsize=(14, 7))
cs = pimprov_truthStateOrigP_dailyRunoff_rmse.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-20, vmax=20)
cbar = plt.colorbar(cs, extend='both').set_label('Percentage (%)', fontsize=20)
plt.title('PIMPROV(RMSE) of daily surface runoff, truthState_OrigP (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'pimprov_rmse.truthStateOrigP.dailyRunoff.png'), format='png')

# --- PIMPROV(baseflow_truthStateOrigP, RMSE) --- #
fig = plt.figure(figsize=(14, 7))
cs = pimprov_truthStateOrigP_dailyBaseflow_rmse.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-100, vmax=100)
cbar = plt.colorbar(cs).set_label('Percentage (%)', fontsize=20)
plt.title('PIMPROV(RMSE) of daily baseflow, truthState_OrigP (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'pimprov_rmse.truthStateOrigP.dailyBaseflow.png'), format='png')

# --- PBIAS(sm1_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_pbias_openloop_sm1.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-20, vmax=20)
cbar = plt.colorbar(cs, extend='both').set_label('PBIAS (%)', fontsize=20)
plt.title('PBIAS of sm1 state, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir, 'pbias.openloop.sm1.png'), format='png')

# --- PBIAS(sm2_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_pbias_openloop_sm2.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-20, vmax=20)
cbar = plt.colorbar(cs, extend='both').set_label('PBIAS (%)', fontsize=20)
plt.title('PBIAS of sm2 state, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir, 'pbias.openloop.sm2.png'), format='png')

# --- PBIAS(sm3_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_pbias_openloop_sm3.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-100, vmax=100)
cbar = plt.colorbar(cs, extend='both').set_label('PBIAS (%)', fontsize=20)
plt.title('PBIAS of sm3 state, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir, 'pbias.openloop.sm3.png'), format='png')

# --- PBIAS(runoff_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_pbias_openloop_dailyRunoff.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-50, vmax=50)
cbar = plt.colorbar(cs, extend='both').set_label('PBIAS (%)', fontsize=20)
plt.title('PBIAS of daily surface runoff, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'pbias.openloop.dailyRunoff.png'), format='png')

# --- PBIAS(baseflow_openloop) --- #
fig = plt.figure(figsize=(14, 7))
cs = da_pbias_openloop_dailyBaseflow.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-100, vmax=100)
cbar = plt.colorbar(cs, extend='both').set_label('PBIAS (%)', fontsize=20)
plt.title('PBIAS of daily baseflow, open-loop (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'pbias.openloop.dailyBaseflow.png'), format='png')

# --- PIMPROV(runoff_truthStateOrigP, PBIAS) --- #
fig = plt.figure(figsize=(14, 7))
cs = pimprov_truthStateOrigP_dailyRunoff_pbias.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-100, vmax=100)
cbar = plt.colorbar(cs, extend='both').set_label('Percentage (%)', fontsize=20)
plt.title('PIMPROV(PBIAS) of daily surface runoff, truthState_OrigP (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'pimprov_pbias.truthStateOrigP.dailyRunoff.png'), format='png')

# --- PIMPROV(baseflow_truthStateOrigP, PBIAS) --- #
fig = plt.figure(figsize=(14, 7))
cs = pimprov_truthStateOrigP_dailyBaseflow_pbias.where(da_domain==1).plot(
    add_colorbar=False, cmap='RdBu',
    vmin=-100, vmax=100)
cbar = plt.colorbar(cs, extend='both').set_label('Percentage (%)', fontsize=20)
plt.title('PIMPROV(PBIAS) of daily baseflow, truthState_OrigP (1980-1989)', fontsize=20)
fig.savefig(os.path.join(output_maps_dir,
                         'pimprov_pbias.truthStateOrigP.dailyBaseflow.png'), format='png')




