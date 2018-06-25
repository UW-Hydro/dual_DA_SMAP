
import xarray as xr
import os


# ============================================================== #
# Parameter setting
# ============================================================== #
orig_forcing_dir = './'  # file name: conus_ens_{XXX}.nc
domain_file = '/civil/hydro/ymao/data_assim/param/vic/ArkRed/ArkRed.domain.nc'

# ============================================================== #
# Load data
# ============================================================== #
ds = xr.open_dataset(os.path.join(orig_forcing_dir, 'conus_ens_100.nc'))
ds_domain = xr.open_dataset(domain_file)

# ============================================================== #
# Subset domain
# ============================================================== #
lat_min = ds_domain['lat'].values.min()
lat_max = ds_domain['lat'].values.max()
lon_min = ds_domain['lon'].values.min()
lon_max = ds_domain['lon'].values.max()

print(ds)
exit()

ds_small = ds.sel(latitude=slice(lat_min, lat_max),
                  longitude=slice(lon_min, lon_max))

print(ds_small)





