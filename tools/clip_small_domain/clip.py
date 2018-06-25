
# This script simply clips a subdomain (rectangular) from a netCDF file. Will not deal with active/inactive cells.

import xarray as xr
import sys
import subprocess

# --- Process input arguments --- #
target_domain_nc = sys.argv[1]  # will use the lat lon dimensions to clip
input_nc = sys.argv[2]
output_nc = sys.argv[3]

# --- Load domain file --- #
ds_domain = xr.open_dataset(target_domain_nc)
lat_min = ds_domain['lat'].values.min()
lat_max = ds_domain['lat'].values.max()
lon_min = ds_domain['lon'].values.min()
lon_max = ds_domain['lon'].values.max()

# --- Use ncks to clip --- #
subprocess.call(
    "ncks -O -d lat,{},{} -d lon,{},{} {} {}".format(
        lat_min-0.001, lat_max+0.001, lon_min-0.001, lon_max+0.001,
        input_nc, output_nc),
    shell=True)

