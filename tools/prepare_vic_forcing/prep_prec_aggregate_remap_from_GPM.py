
# This script:
#   1) aggregates 30-min GPM precipitation data to a specified timestep;
#   2) remap it to a specified domain. The domain can be a VIC-domain, or another domain
#   3) output the resulting precipitation data


import sys
import pandas as pd
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

from tonic.io import read_config, read_configobj
from prep_forcing_utils import (to_netcdf_forcing_file_compress, setup_output_dirs,
                                remap_con, add_gridlines, edges_from_centers)


# ======================================================= #
# Process command line argument
# ======================================================= #
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_dir = cfg['OUTPUT']['out_dir']

output_subdir_plots = setup_output_dirs(output_dir, mkdirs=['plots'])['plots']
output_subdir_data_remapped = setup_output_dirs(
    output_dir, mkdirs=['prec_only'])['prec_only']
output_subdir_tmp = setup_output_dirs(output_dir, mkdirs=['tmp'])['tmp']


# ======================================================= #
# Process data for each VIC timestep
# ======================================================= #
# --- Loop over each VIC timestep (if > hourly, each VIC timestep --- #
# --- will contain multiple hourly NLDAS-2 data) --- #
vic_times = pd.date_range(
    cfg['TIME']['start_time'],
    cfg['TIME']['end_time'],
    freq='{}H'.format(cfg['VIC']['time_step']))

# Initialize a dictionary to store VIC-timestep forcing data
# Each element in the dict will be a list for one year
start_year = vic_times[0].year
end_year = vic_times[-1].year
dict_force_yearly_vic_timestep = {}  # This dict will save forcing data
dict_times_yearly_vic_timestep = {}  # This dict will save time indices
for year in range(start_year, end_year+1):
    dict_force_yearly_vic_timestep[year] = []
    dict_times_yearly_vic_timestep[year] = []
    
# --- Loop over each VIC timestep --- #
for vic_time in vic_times:
    # --- Loop over each 30-min in this VIC timestep and load NLDAS hourly data --- #
    list_ds = []
    for hour in range(0, int(cfg['VIC']['time_step'])):
        for minute in [0, 30]:
            time = vic_time + pd.DateOffset(seconds=3600*hour+60*minute)
            filename = os.path.join(
                cfg['GPM']['gpm_dir'],
                '{}'.format(time.year), '{:02d}'.format(time.month),
                '{:02d}'.format(time.day),
                '{}.{:02d}00.nc'.format(time.strftime('%Y%m%d'), time.hour))
            # Load data
            ds = xr.open_dataset(filename)
            list_ds.append(ds)
    ds_vic_time = xr.concat(list_ds, dim='time')
    # --- Aggregate each variable --- #
    # Precipitation - sum; [mm/hour] -> [mm/step]
    da_prec = ds_vic_time['precipitationUncal'].mean(dim='time') * cfg['VIC']['time_step']
    da_prec.attrs['unit'] = 'mm/step'
    da_prec = da_prec.transpose('lat', 'lon')
    # Save data to file
    ds_force_vic_time = xr.Dataset(
        {'PREC': da_prec})
    # Store dataset to the list of corresponding year
    current_year = vic_time.year
    dict_force_yearly_vic_timestep[current_year].append(ds_force_vic_time)
    dict_times_yearly_vic_timestep[current_year].append(vic_time)


# ======================================================= #
# Concat each year of forcing
# ======================================================= #
ds_force_yearly = {}  # Dict keys are year
for year in range(start_year, end_year+1):
    list_force_vic_timestep = dict_force_yearly_vic_timestep[year]
    ds_concat_yearly = xr.concat(list_force_vic_timestep, dim='time')
    ds_concat_yearly['time'] = dict_times_yearly_vic_timestep[year]
    ds_force_yearly[year] = ds_concat_yearly


# ============================================================ #
# Double check GPM domain compared to VIC domain by plotting
# (GPM domain should cover the entire VIC domain)
# ============================================================ #
# --- Load VIC domain --- #
ds_vic_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])
da_vic_domain = ds_vic_domain[cfg['DOMAIN']['mask_name']]
# --- Extract GPM domain --- #
da_gpm_domain = xr.DataArray(np.ones([len(ds_force_yearly[start_year]['lat']),
                                      len(ds_force_yearly[start_year]['lon'])],
                                     dtype=int),
                             coords=[ds_force_yearly[start_year]['lat'], ds_force_yearly[start_year]['lon']],
                             dims=['lat', 'lon'])
# --- Plot VIC and GPM domain to check --- #
fig = plt.figure(figsize=(16, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([float(da_gpm_domain['lon'].min().values) - 0.5,
               float(da_gpm_domain['lon'].max().values) + 0.5,
               float(da_gpm_domain['lat'].min().values) - 0.5,
               float(da_gpm_domain['lat'].max().values) + 0.5], ccrs.Geodetic())
gl = add_gridlines(
    ax,
    xlocs=np.arange(float(da_gpm_domain['lon'].min().values) -0.5,
                    float(da_gpm_domain['lon'].max().values) +0.5, 1),
                    ylocs=np.arange(
                        float(da_gpm_domain['lat'].min().values) - 0.5,
                        float(da_gpm_domain['lat'].max().values) + 0.5, 1),alpha=0)
# Plot GPM grids
lon_edges = edges_from_centers(da_gpm_domain['lon'].values)
lat_edges = edges_from_centers(da_gpm_domain['lat'].values)
lonlon, latlat = np.meshgrid(lon_edges, lat_edges)
cs = plt.pcolormesh(
    lonlon, latlat, da_gpm_domain.values,
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
fig.savefig(os.path.join(output_subdir_plots, 'gpm_domain_check.png'),
            format='png')


# ======================================================= #
# Remap to the VIC parameter grid
# (using conservative remapping)
# ======================================================= #
print('Remapping GPM...')
# --- Remap GPM to VIC grid for each year --- #
dict_gpm_remapped = {}  # key: year; item: remapped GPM
for year in range(start_year, end_year+1):
    print('\t', year)
    if year == start_year:  # only calculate weights for the first year
        # If specify weight file already, directly use it
        if 'REMAP' in cfg and 'weight_nc' in cfg['REMAP']:
            reuse_weight = True
            final_weight_nc = cfg['REMAP']['weight_nc']
        # Else, calculate weights
        else:
            reuse_weight = False
            final_weight_nc = os.path.join(output_subdir_tmp, 'gpm_to_vic_weights.nc')
        da_gpm_remapped, weight_array = remap_con(
            reuse_weight=reuse_weight,
            da_source=ds_force_yearly[year]['PREC'],
            final_weight_nc=final_weight_nc,
            da_source_domain=da_gpm_domain,
            da_target_domain=da_vic_domain,
            tmp_weight_nc=os.path.join(output_subdir_tmp, 'gpm_to_vic_weights.tmp.nc'),
            process_method=None)
    else:
        da_gpm_remapped, weight_array = remap_con(
            reuse_weight=True,
            da_source=ds_force_yearly[year]['PREC'],
            final_weight_nc=final_weight_nc,
            da_source_domain=da_gpm_domain,
            da_target_domain=da_vic_domain,
            tmp_weight_nc=os.path.join(output_subdir_tmp, 'gpm_to_vic_weights.tmp.nc'),
            process_method=None)
    dict_gpm_remapped[year] = da_gpm_remapped


# ======================================================= #
# Mask forcing data with domain file
# ======================================================= #
# --- Load domain file --- #
ds_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])
mask_var = cfg['DOMAIN']['mask_name']
lat_name = cfg['DOMAIN']['lat_name']
lon_name = cfg['DOMAIN']['lon_name']
# --- Mask --- #
for year in range(start_year, end_year+1):
    dict_gpm_remapped[year] = dict_gpm_remapped[year].where(ds_domain[mask_var].values)


# ======================================================= #
# Save final forcing data to file
# ======================================================= #
for year in range(start_year, end_year+1):
    ds = xr.Dataset({'PREC': dict_gpm_remapped[year]})
    to_netcdf_forcing_file_compress(
        ds,
        os.path.join(output_subdir_data_remapped, 'force.{}.nc'.format(year)))





