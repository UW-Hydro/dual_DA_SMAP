
import xarray as xr
import pandas as pd
import os
import sys


# ====================================================== #
# Parameters
# ====================================================== #
# --- Orig. Newman forcing file directory (file names: conus_ens_XXX.nc) --- #
orig_dir = '/civil/hydro/ymao/data_assim/data/Newman_ensemble_forcing'

# --- Domain file (domain to extract) --- #
# Domain netCDF file
domain_nc = '/civil/hydro/ymao/data_assim/param/vic/ArkRed/ArkRed.domain.nc'
# Varname of mask in the domain file
mask_varname = 'mask'

# --- Time period to extract --- #
start_date = pd.datetime(1980, 1, 1)
end_date = pd.datetime(1989, 12, 31)

# --- Ensemble members to process --- #
ens_to_process = [int(sys.argv[1])]

# --- Output --- #
# Output directory for extracted netCDF files
out_nc_dir = '/civil/hydro/ymao/data_assim/data/Newman_ensemble_forcing/ArkRed.1980_1989'

# ====================================================== #
# Load in domain file
# ====================================================== #
ds_domain = xr.open_dataset(domain_nc)
da_domain = ds_domain[mask_varname]

lat_min = da_domain['lat'].min().values
lat_max = da_domain['lat'].max().values
lon_min = da_domain['lon'].min().values
lon_max = da_domain['lon'].max().values


# ====================================================== #
# Extract domain and time period for all ensemble members
# ====================================================== #

# Loop over each enemble member
for i in ens_to_process:

	print('Processing ensemble member {}...'.format(i))

	# --- Load orig. forcing data file --- #
	filename = os.path.join(orig_dir, 'conus_ens_{:03d}.nc'.format(i))
	ds = xr.open_dataset(filename)

	# --- Select out time period needed --- #
	ds = ds.sel(time=slice(start_date, end_date))

	# --- Rename lat lon dimensions --- #
	lat = ds['latitude'].values[:, 0]
	lon = ds['longitude'].values[0, :]
	elev_attrs = ds['elevation'].attrs
	pcp_attrs = ds['pcp'].attrs
	t_mean_attrs = ds['t_mean'].attrs
	t_range_attrs = ds['t_range'].attrs

	ds = xr.Dataset({'elevation': (['lat', 'lon'], ds['elevation'].values),
                     'pcp': (['time', 'lat', 'lon'], ds['pcp'].values),
                     't_mean': (['time', 'lat', 'lon'], ds['t_mean'].values),
                     't_range': (['time', 'lat', 'lon'], ds['t_range'].values)},
                    coords={'lat': lat, 'lon': lon, 'time': ds['time'].values})

	# --- Extract domain --- #
	ds_small = ds.sel(lat=slice(lat_min, lat_max),
                      lon=slice(lon_min, lon_max))
	ds_masked = ds_small.where(da_domain.values)

	# --- Put attributes back --- #
	ds_masked['elevation'].attrs = elev_attrs
	ds_masked['pcp'].attrs = pcp_attrs
	ds_masked['t_mean'].attrs = t_mean_attrs
	ds_masked['t_range'].attrs = t_range_attrs

	# --- Write out a new netCDF file --- #
	ds_masked.to_netcdf(os.path.join(out_nc_dir, 'ens_{:03d}.nc'.format(i)),
						format='NETCDF4_CLASSIC')


