import xarray as xr
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import h5py
import datetime as dt
import glob
import os
from scipy.stats import rankdata
import sys

from tonic.io import read_config, read_configobj
from da_utils import (setup_output_dirs, calculate_smap_domain_from_vic_domain,
                      extract_smap_static_info, extract_smap_sm,
                      extract_smap_multiple_days, edges_from_centers, add_gridlines,
                      find_global_param_value, remap_con, rescale_SMAP_domain)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Parameter setting
# ============================================================ #
start_date = pd.to_datetime(cfg['TIME']['start_date'])
end_date = pd.to_datetime(cfg['TIME']['end_date'])

output_dir = cfg['OUTPUT']['output_dir']


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_subdir_plots = setup_output_dirs(output_dir, mkdirs=['plots'])['plots']
output_subdir_data_unscaled = setup_output_dirs(output_dir, mkdirs=['data_unscaled'])['data_unscaled']
output_subdir_data_scaled = setup_output_dirs(output_dir, mkdirs=['data_scaled'])['data_scaled']
output_subdir_tmp = setup_output_dirs(output_dir, mkdirs=['tmp'])['tmp']


# ============================================================ #
# Determine SMAP domain needed based on VIC domain
# ============================================================ #
print('Determing SMAP domain...')
# --- Load VIC domain --- #
ds_vic_domain = xr.open_dataset(cfg['DOMAIN']['vic_domain_nc'])
da_vic_domain = ds_vic_domain[cfg['DOMAIN']['mask_name']]
# --- Load one example SMAP file --- #
da_smap_example = extract_smap_multiple_days(
    os.path.join(cfg['INPUT']['smap_dir'], 'SMAP_L3_SM_P_{}_*.h5'),
    start_date.strftime('%Y%m%d'), start_date.strftime('%Y%m%d'))
# --- Calculate SMAP domain needed --- #
da_smap_domain = calculate_smap_domain_from_vic_domain(da_vic_domain, da_smap_example)

# --- Plot VIC and SMAP domain to check --- #
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([float(da_smap_domain['lon'].min().values) - 0.5,
               float(da_smap_domain['lon'].max().values) + 0.5,
               float(da_smap_domain['lat'].min().values) - 0.5,
               float(da_smap_domain['lat'].max().values) + 0.5], ccrs.Geodetic())
gl = add_gridlines(
    ax,
    xlocs=np.arange(float(da_smap_domain['lon'].min().values) -0.5,
                    float(da_smap_domain['lon'].max().values) +0.5, 1),
                    ylocs=np.arange(
                        float(da_smap_domain['lat'].min().values) - 0.5,
                        float(da_smap_domain['lat'].max().values) + 0.5, 1),alpha=0)
# Plot SMAP grids
lon_edges = edges_from_centers(da_smap_domain['lon'].values)
lat_edges = edges_from_centers(da_smap_domain['lat'].values)
lonlon, latlat = np.meshgrid(lon_edges, lat_edges)
cs = plt.pcolormesh(
    lonlon, latlat, da_smap_domain.values,
    cmap='Spectral',
    vmin=0, vmax=1,
    transform=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor=(1, 0, 0, 0.5))
# Plot 1/8th grid cell centers
lon_edges = edges_from_centers(da_vic_domain['lon'].values)
lat_edges = edges_from_centers(da_vic_domain['lat'].values)
lonlon, latlat = np.meshgrid(lon_edges, lat_edges)
cs = plt.pcolormesh(
    lonlon, latlat, da_vic_domain.values,
    cmap='Reds',
    vmin=0, vmax=1,
    transform=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor=(1, 0, 0, 0.5))
# Make plot looks better
ax.coastlines()
# Save figure
fig.savefig(os.path.join(output_subdir_plots, 'smap_domain_check.png'),
            format='png')


# ============================================================ #
# Load and process SMAP data
# ============================================================ #
print('Loading and processing SMAP data...')
# --- Load data --- #
print('Extracting SMAP data')
# If SMAP data is already processed before, directly load
if cfg['INPUT']['smap_exist'] is True:
    # --- Load processed SMAP data --- #
    da_smap = xr.open_dataset(cfg['INPUT']['smap_unscaled_nc'])['soil_moisture']
    # --- Extract AM and PM time points --- #
    shift_hours = int(cfg['TIME']['smap_shift_hours'])
    # AM
    am_hour = (6 + shift_hours) if (6 + shift_hours) < 24 else (6 + shift_hours - 24)
    smap_times_am_ind = np.asarray([pd.to_datetime(t).hour==am_hour
                                    for t in da_smap['time'].values])
    smap_times_am = da_smap['time'].values[smap_times_am_ind]
    # PM
    pm_hour = (18 + shift_hours) if (18 + shift_hours) < 24 else (18 + shift_hours - 24)
    smap_times_pm_ind = np.asarray([pd.to_datetime(t).hour==pm_hour
                                    for t in da_smap['time'].values])
    smap_times_pm = da_smap['time'].values[smap_times_pm_ind]

# If SMAP data not processed, before, load and process
else:
    # --- Load SMAP data --- #
    da_smap = extract_smap_multiple_days(
        os.path.join(cfg['INPUT']['smap_dir'], 'SMAP_L3_SM_P_{}_*.h5'),
        start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'),
        da_smap_domain=da_smap_domain)
    # --- Convert SMAP time to VIC-forcing-data time zone --- #
    # --- Shift SMAP data to the VIC-forcing-data time zone --- #
    # Shift SMAP time
    shift_hours = int(cfg['TIME']['smap_shift_hours'])
    smap_times_shifted = \
        [pd.to_datetime(t) + pd.DateOffset(seconds=3600*shift_hours)
         for t in da_smap['time'].values]
    da_smap['time'] = smap_times_shifted
    # --- Exclude SMAP data points after shifting that are outside of the processing time period --- #
    da_smap = da_smap.sel(
        time=slice(start_date.strftime('%Y%m%d')+'-00',
                   end_date.strftime('%Y%m%d')+'-23'))
    # --- Get a list of SMAP AM & PM time points after shifting --- #
    # AM
    am_hour = (6 + shift_hours) if (6 + shift_hours) < 24 else (6 + shift_hours - 24)
    smap_times_am_ind = np.asarray([pd.to_datetime(t).hour==am_hour
                                    for t in da_smap['time'].values])
    smap_times_am = da_smap['time'].values[smap_times_am_ind]
    # PM
    pm_hour = (18 + shift_hours) if (18 + shift_hours) < 24 else (18 + shift_hours - 24)
    smap_times_pm_ind = np.asarray([pd.to_datetime(t).hour==pm_hour
                                    for t in da_smap['time'].values])
    smap_times_pm = da_smap['time'].values[smap_times_pm_ind]
    # --- Save processed SMAP data to file --- #
    ds_smap = xr.Dataset({'soil_moisture': da_smap})
    ds_smap.to_netcdf(
        os.path.join(output_subdir_data_unscaled,
                     'soil_moisture_unscaled.{}_{}.nc'.format(
                        start_date.strftime('%Y%m%d'),
                        end_date.strftime('%Y%m%d'))))


# ============================================================ #
# SMAP data quality control
# ============================================================ #
print('Quality control...')
if cfg['QC']['qc_method'] == 'no_winter':  # If exclude Nov - Feb data
    for t in da_smap['time'].values:
        if pd.to_datetime(t).month in [11, 12, 1, 2]:
            da_smap.loc[t, :, :] = np.nan
    ds_smap = xr.Dataset({'soil_moisture': da_smap})
    ds_smap.to_netcdf(
        os.path.join(output_subdir_data_unscaled,
                     'soil_moisture_unscaled.qc_{}.{}_{}.nc'.format(
                        cfg['QC']['qc_method'],
                        start_date.strftime('%Y%m%d'),
                        end_date.strftime('%Y%m%d'))))


# ============================================================ #
# Rescale SMAP to the VIC regime
# ============================================================ #
print('Rescaling SMAP...')
# --- Load reference VIC history file --- #
ds_vic_hist = xr.open_dataset(cfg['RESCALE']['vic_history_nc'])

# --- Extract the domain and period to be consistent with SMAP data --- #
ds_vic_hist = ds_vic_hist.sel(
    lat=slice(da_vic_domain['lat'].values[0]-0.05, da_vic_domain['lat'].values[-1]+0.05),
    lon=slice(da_vic_domain['lon'].values[0]-0.05, da_vic_domain['lon'].values[-1]+0.05),
    time=slice(start_date.strftime('%Y%m%d')+'-00',
               end_date.strftime('%Y%m%d')+'-23'))

# --- Extract VIC surface soil moisture at time steps matching SMAP AM & PM --- #
# Shift the VIC soil moisture time to the correct time point
# (since the SOIL moisture output is timestep-end)
vic_model_steps_per_day = cfg['RESCALE']['vic_model_steps_per_day']
vic_timestep = int(24 / vic_model_steps_per_day)  # [hour]
vic_sm_times = [pd.to_datetime(t) + pd.DateOffset(hours=vic_timestep) for t in ds_vic_hist['time'].values]
da_vic_sm = ds_vic_hist['OUT_SOIL_MOIST'].sel(nlayer=0).copy(deep=True)
da_vic_sm['time'] = vic_sm_times

# --- Remap VIC surface SM data to SMAP grid cell resolution --- #
if cfg['RESCALE']['reuse_weight']:
    da_vic_remapped, weight_array = remap_con(
        reuse_weight=True,
        da_source=da_vic_sm,
        final_weight_nc=cfg['RESCALE']['weight_nc'],
        da_source_domain=da_vic_domain,
        da_target_domain=da_smap_domain,
        tmp_weight_nc=os.path.join(output_subdir_tmp, 'vic_to_smap_weights.tmp.nc'),
        process_method=None)
else:
    da_vic_remapped, weight_array = remap_con(
        reuse_weight=False,
        da_source=da_vic_sm,
        final_weight_nc=os.path.join(output_subdir_tmp, 'vic_to_smap_weights.nc'),
        da_source_domain=da_vic_domain,
        da_target_domain=da_smap_domain,
        tmp_weight_nc=os.path.join(output_subdir_tmp, 'vic_to_smap_weights.tmp.nc'),
        process_method=None)

# --- Rescale SMAP data (for AM and PM seperately) --- #
# Load unscaled measurement error domain and convert to [time, lat, lon]
da_meas_error_unscaled = xr.open_dataset(
    cfg['INPUT']['meas_error_unscaled_nc'])[cfg['INPUT']['meas_error_unscaled_varname']]
# Rescale
da_smap_rescaled, da_meas_error_rescaled = rescale_SMAP_domain(da_smap, da_vic_remapped,
                    smap_times_am, smap_times_pm,
                    da_meas_error_unscaled,
                    method=cfg['RESCALE']['rescale_method'])

# --- Save rescaled SMAP data to file --- #
ds_smap_rescaled = xr.Dataset({'soil_moisture': da_smap_rescaled})
ds_smap_rescaled.to_netcdf(
    os.path.join(output_subdir_data_scaled,
                 'soil_moisture_scaled.{}.{}{}_{}.nc'.format(
                     cfg['RESCALE']['rescale_method'],
                     cfg['QC']['qc_method']+'.' if cfg['QC']['qc_method'] is not None else '',
                     start_date.strftime('%Y%m%d'),
                     end_date.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

# --- Save rescaled measurement error to file --- #
da_meas_error_rescaled.attrs['unit'] = 'mm'
ds_meas_error_rescaled = xr.Dataset({'soil_moisture_error': da_meas_error_rescaled})
ds_meas_error_rescaled.to_netcdf(
    os.path.join(output_subdir_data_scaled,
                 'soil_moisture_error_scaled.{}.{}{}_{}.nc'.format(
                     cfg['RESCALE']['rescale_method'],
                     cfg['QC']['qc_method']+'.' if cfg['QC']['qc_method'] is not None else '',
                     start_date.strftime('%Y%m%d'),
                     end_date.strftime('%Y%m%d'))),
    format='NETCDF4_CLASSIC')

# ============================================================ #
# Perhaps plot some check plots
# ============================================================ #


