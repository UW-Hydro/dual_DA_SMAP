
import sys
import pandas as pd
import os
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tonic.io import read_config, read_configobj
from prep_forcing_utils import to_netcdf_forcing_file_compress, setup_output_dirs


# ======================================================= #
# Process command line argument
# ======================================================= #
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Setup output subdirs
# ============================================================ #
output_dir = cfg['OUTPUT']['out_dir']

output_subdir_data_prec = setup_output_dirs(
    output_dir, mkdirs=['prec_only'])['prec_only']
output_subdir_data_force = setup_output_dirs(
    output_dir, mkdirs=['force_with_NLDAS2'])['force_with_NLDAS2']


# ======================================================= #
# For each year, load GPM and NLDAS-2 forcing data and put together, and save to file
# ======================================================= #
# Initialize a dictionary to store final forcing data
start_time = pd.to_datetime(cfg['TIME']['start_time'])
end_time = pd.to_datetime(cfg['TIME']['end_time'])
start_year = start_time.year
end_year = end_time.year
dict_force_yearly = {}  # This dict will save forcing data
for year in range(start_year, end_year+1):
    # Load GPM data
    da_gpm = xr.open_dataset(
        os.path.join(output_subdir_data_prec,
                     'force.{}.nc'.format(year)))['PREC'].sel(
        time=slice(start_time, end_time))
    # Load NLDAS-2 data
    ds_nldas2 = xr.open_dataset('{}{}.nc'.format(
        cfg['NLDAS2']['force_basedir'], year)).sel(
            time=slice(start_time, end_time))
    ds_nldas2.load()
    ds_nldas2.close()
    # Replace PREC by GPM data
    ds_nldas2['PREC'] = da_gpm
    # Save final forcing file
    to_netcdf_forcing_file_compress(
        ds_nldas2,
        os.path.join(output_subdir_data_force, 'force.{}.nc'.format(year)))



