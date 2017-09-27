
''' This script:
        - converts VIC4 ascii format parameter files to netCDF format;
        - subset netCDF param files to a smaller target domain

    Usage:
       $ python prep_vic_nc_params.py config_file
'''

import xarray as xr
import os
import getpass
from datetime import datetime
import numpy as np
import subprocess
import sys

from tonic.io import read_config, read_configobj
import tonic.models.vic.grid_params as gp

# Metadata to be used later
user = getpass.getuser()
now = datetime.now()


# ============================================================== #
# Load in config file
# ============================================================== #
cfg = read_configobj(sys.argv[1])


# ============================================================== #
# Convert VIC parameter files to netCDF - same domain as ascii param files
# ============================================================== #

soil_file = cfg['PARAM']['soil_asc']
snow_file = cfg['PARAM']['snowband_asc']
veg_file = cfg['PARAM']['vegparam_asc']
vegl_file = cfg['PARAM']['veglib_asc']

out_file = os.path.join(cfg['OUTPUT']['out_param_dir'],
                        '{}.param.nc'.format(cfg['OUTPUT']['orig_asc_domain_name']))
                        
cols = gp.Cols(nlayers=cfg['PARAM']['nlayers'],
               snow_bands=cfg['PARAM']['snow_bands'],
               organic_fract=cfg['PARAM']['organic_fract'],
               spatial_frost=cfg['PARAM']['spatial_frost'],
               spatial_snow=cfg['PARAM']['spatial_snow'],
               july_tavg_supplied=cfg['PARAM']['july_tavg_supplied'],
               veglib_fcan=cfg['PARAM']['veglib_fcan'],
               veglib_photo=cfg['PARAM']['veglib_photo'])
n_veg_classes = cfg['PARAM']['n_veg_classes']
root_zones = cfg['PARAM']['root_zones']
vegparam_lai = cfg['PARAM']['vegparam_lai']
lai_src = cfg['PARAM']['lai_src']

# ----------------------------------------------------------------- #

# Read the soil parameters
soil_dict = gp.soil(soil_file, c=cols)

# Read the snow parameters
snow_dict = gp.snow(snow_file, soil_dict, c=cols)

# Read the veg parameter file
veg_dict = gp.veg(veg_file, soil_dict,
                  vegparam_lai=vegparam_lai, lai_src=lai_src, 
                  veg_classes=n_veg_classes, max_roots=root_zones)

# Read the veg library file
veg_lib, lib_bare_idx = gp.veg_class(vegl_file, c=cols)

# Determine the grid shape
target_grid, target_attrs = gp.calc_grid(soil_dict['lats'], soil_dict['lons'])

# Grid all the parameters
grid_dict = gp.grid_params(soil_dict, target_grid, version_in='4', 
                           vegparam_lai=vegparam_lai, lai_src=lai_src,
                           lib_bare_idx=lib_bare_idx, 
                           veg_dict=veg_dict, veglib_dict=veg_lib, 
                           snow_dict=snow_dict, lake_dict=None)

# Write a netCDF file with all the parameters
gp.write_netcdf(out_file,
                target_attrs,
                target_grid=target_grid,
                vegparam_lai=vegparam_lai,
                lai_src=lai_src,
                soil_grid=grid_dict['soil_dict'],
                snow_grid=grid_dict['snow_dict'],
                veg_grid=grid_dict['veg_dict'])


# ============================================================== #
# Make domain file
# ============================================================== #
dom_ds = xr.Dataset()

# Set global attributes
dom_ds.attrs['title'] = 'VIC domain data'
dom_ds.attrs['Conventions'] = 'CF-1.6'
dom_ds.attrs['history'] = 'created by %s, %s' % (user, now)
dom_ds.attrs['user_comment'] = 'VIC domain data'
dom_ds.attrs['source'] = 'generated from VIC CONUS 1.8 deg model parameters, see Maurer et al. (2002) for more information'

dom_file = os.path.join(cfg['OUTPUT']['out_param_dir'],
                        '{}.domain.nc'.format(cfg['OUTPUT']['target_domain_name']))

# Load the mask variable
da_mask = xr.open_dataset(cfg['MASK']['mask_nc'])\
          [cfg['MASK']['mask_varname']]
dom_ds['mask'] = da_mask

# For now, the frac variable is going to be just like the mask
dom_ds['frac'] = dom_ds['mask'].astype(np.float)
dom_ds['frac'].attrs['long_name'] = 'fraction of grid cell that is active'
dom_ds['frac'].attrs['units'] = '1'

# Set variable attributes
for k, v in target_attrs.items():
    if k == 'xc':
        k = 'lon'
    elif k == 'yc':
        k = 'lat'
    dom_ds[k].attrs = v
    
# Write temporary file for gridarea calculation
dom_ds.to_netcdf(os.path.join(cfg['OUTPUT']['out_param_dir'], 'temp.nc'),
                 format='NETCDF4_CLASSIC')

# Calculate grid area
subprocess.call("cdo gridarea {} {}".format(
                    os.path.join(cfg['OUTPUT']['out_param_dir'], 'temp.nc'),
                    os.path.join(cfg['OUTPUT']['out_param_dir'], 'area.nc')),
                shell=True)

# Extract the area variable
area = xr.open_dataset(os.path.join(cfg['OUTPUT']['out_param_dir'], 'area.nc'))\
                       ['cell_area']
dom_ds['area'] = area

# Write the domain file
dom_ds.to_netcdf(dom_file, format='NETCDF4_CLASSIC')
dom_ds.close()

# Clean up
subprocess.call("rm {} {}".format(
                    os.path.join(cfg['OUTPUT']['out_param_dir'], 'temp.nc'),
                    os.path.join(cfg['OUTPUT']['out_param_dir'], 'area.nc')),
                shell=True)


# ============================================================== #
# Extract param file for target domain
# ============================================================== #

# Load in orig-whole-domain param file
ds_param_orig = xr.open_dataset(out_file, decode_cf=False)

# Subset the orig param file
lat_min = da_mask['lat'].min().values
lat_max = da_mask['lat'].max().values
lon_min = da_mask['lon'].min().values
lon_max = da_mask['lon'].max().values
ds_param_target_domain = ds_param_orig.sel(lat=slice(lat_min, lat_max),
                                           lon=slice(lon_min, lon_max))

# Mask "mask" variable
mask = (da_mask>0).values

mask_array = ds_param_target_domain['mask'].values
mask_array[~mask] = 0
ds_param_target_domain['mask'][:] = mask_array

# Mask "run_cell" variable
run_cell = ds_param_target_domain['run_cell'].values
run_cell[~mask] = 0
ds_param_target_domain['run_cell'][:] = run_cell

# Save to netCDF file
ds_param_target_domain.to_netcdf(
    os.path.join(cfg['OUTPUT']['out_param_dir'],
                 '{}.param.nc'.format(cfg['OUTPUT']['target_domain_name'])),
    format='NETCDF4_CLASSIC')

