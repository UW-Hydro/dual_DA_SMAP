
# Extract SMAP domain needed and weight file

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
                      find_global_param_value, remap_con, rescale_SMAP_domain,
                      change_weights_no_split)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Determine SMAP domain needed based on VIC domain
# ============================================================ #
print('Determing SMAP domain...')
# --- Load VIC domain --- #
ds_vic_domain = xr.open_dataset(cfg['DOMAIN']['vic_domain_nc'])
da_vic_domain = ds_vic_domain[cfg['DOMAIN']['mask_name']]
# --- Load big SMAP domain --- #
da_smap_domain = xr.open_dataset(cfg['INPUT']['smap_domain'])
# --- Calculate SMAP domain needed and save --- #
da_smap_domain = calculate_smap_domain_from_vic_domain(
    da_vic_domain, da_smap_domain, is_smap_domain=True)
ds_smap_domain = xr.Dataset({'mask': da_smap_domain})
ds_smap_domain.to_netcdf(
    os.path.join(cfg['OUTPUT']['output_dir'], 'domain.smap.nc'),
    format='NETCDF4_CLASSIC')

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
fig.savefig(os.path.join(cfg['OUTPUT']['output_dir'], 'check_plot.smap_domain.png'),
            format='png')


# ============================================================ #
# Calculate weight file
# ============================================================ #
# --- Calculate and save conservative weight --- #
da_vic_remapped, weight_array = remap_con(
    reuse_weight=False,
    da_source=da_vic_domain,
    final_weight_nc=os.path.join(cfg['OUTPUT']['output_dir'], 'vic_to_smap_weights.nc'),
    da_source_domain=da_vic_domain,
    da_target_domain=da_smap_domain,
    tmp_weight_nc=os.path.join(cfg['OUTPUT']['output_dir'], 'vic_to_smap_weights.tmp.nc'),
    process_method=None)

# --- Change weight to be no-split-cell and save --- #
change_weights_no_split(
    weight_orig_nc=os.path.join(cfg['OUTPUT']['output_dir'], 'vic_to_smap_weights.nc'),
    da_source_domain=da_vic_domain, da_target_domain=da_smap_domain,
    output_weight_nc=os.path.join(cfg['OUTPUT']['output_dir'], 'vic_to_smap_weights.no_split_cell.nc'))





