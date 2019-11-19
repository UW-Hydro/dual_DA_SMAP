
# Process R-generated spatial fields

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import rpy2.robjects as robjects

# ===================================================== #
# Parameters
# ===================================================== #
# Inputs
da_domain = xr.open_dataset(
    '/pool0/data/yixinmao/data_assim/param/vic/ArkRed/ArkRed.domain.nc')['mask']
times = pd.date_range('2015-03-31-00', '2018-01-01-00', freq='12H')
nlayer = 3
nveg = 12
nsnow = 1
R_array_dir = ('/pool0/data/yixinmao/data_assim/tools/prepare_perturbation/'
               'output/R_arrays/phi12.N32')  # File name: time<i>.Rds
N = 32

# Outputs
output_dir = ('/pool0/data/yixinmao/data_assim/tools/prepare_perturbation/'
              'output/random_fields_nc/phi12.N32')


# ===================================================== #
# Load R-generated random fields, process and save
# ===================================================== #
nlat = len(da_domain['lat'])
nlon = len(da_domain['lon'])
n = nlayer * nveg * nsnow

readRDS = robjects.r['readRDS']

for i, t in enumerate(times):
    print(i)
    
    # Initialize
    da_field_from_R = xr.DataArray(
        np.zeros([nlat, nlon, n, N]),
        coords=[da_domain['lat'], da_domain['lon'], range(n), range(N)],
        dims=['lat', 'lon', 'n', 'N'])
    
    # Load R output array
    data = np.array(readRDS(os.path.join(
        R_array_dir, 'time{}.Rds'.format(i+1))))  # [cell, tile*layer*N]
    data = data.reshape([nlat, nlon, n, N])  # zero-mean, unit-variance, vertically uncorrelated
    
    # Put array to da
    da_field_from_R[:] = data
    
    # Save array to file
    ds_field_from_R = xr.Dataset({'noise': da_field_from_R})
    ds_field_from_R.to_netcdf(os.path.join(
        output_dir,
        'vert_uncorr.{}.nc'.format(t.strftime("%Y%m%d-%H-%M-%S"))))


