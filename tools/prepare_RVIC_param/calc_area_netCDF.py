#!/usr/local/anaconda/bin/python

''' This script calculates grid cell area of a domain '''

import xarray as xr
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--domain_file", type=str,  help="A tmp nc file with 'area' variable indicating the domain of the data")
parser.add_argument("--grid_size", type=str,  help="Grid size, e.g., 0.5, 0.25, ...")
args = parser.parse_args()

# Set constants
RERD = 6371229.0  # Radius of earth [m]
grid_size = float(args.grid_size)

# Load domain file and calculate area
ds = xr.open_dataset(args.domain_file)
area = ds.area*0 + RERD*RERD * grid_size/180.0*np.pi * \
            np.absolute(np.sin((ds.lat-grid_size/2.0)/180.0*np.pi) \
                        - np.sin((ds.lat+grid_size/2.0)/180.0*np.pi))

ds['area'] = area
ds['area'].attrs['units'] = 'm2'

# Save as netCDF
ds.to_netcdf(args.domain_file, format='NETCDF4_CLASSIC')


