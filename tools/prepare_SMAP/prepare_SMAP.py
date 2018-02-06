import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import h5py
import datetime as dt
import glob
import os
import sys
from scipy.stats import rankdata

from tonic.io import read_config, read_configobj
from da_utils import (setup_output_dirs, calculate_smap_domain_from_vic_domain,
                      extract_smap_static_info, extract_smap_sm,
                      extract_smap_multiple_days, edges_from_centers, add_gridlines)


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


# ============================================================ #
# Determine SMAP domain needed based on VIC domain
# ============================================================ #
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
# Load SMAP data
# ============================================================ #
# --- Load data --- #
print('Extracting SMAP data')
da_smap = extract_smap_multiple_days(
    os.path.join(cfg['INPUT']['smap_dir'], 'SMAP_L3_SM_P_{}_*.h5'),
    start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'),
    da_smap_domain=da_smap_domain)


# ============================================================ #
# Convert SMAP time to UTC
# ============================================================ #
# !!!!!!!!!!!!!!!!! Still need to do !!!!!!!!!!!!!!!!!!!


# ============================================================ #
# Save processed SMAP data to file
# ============================================================ #
ds_smap = xr.Dataset({'soil_moisture': da_smap})
ds_smap.to_netcdf(
    os.path.join(output_subdir_data_unscaled,
                 'soil_moisture_unscaled.{}_{}.nc'.format(
                     start_date.strftime('%Y%m%d'),
                     end_date.strftime('%Y%m%d'))))

