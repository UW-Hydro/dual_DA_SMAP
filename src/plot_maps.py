

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import os
import sys

from da_utils import rmse, innov_norm_var
from tonic.io import read_config, read_configobj


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ======================================================== #
# Parameter setting
# ======================================================== #
# Time period
EnKF_start_time = pd.to_datetime(cfg['EnKF']['start_time'])
EnKF_end_time = pd.to_datetime(cfg['EnKF']['end_time'])

# For error maps
openloop_hist_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_EnKF_basedir'],
            'history',
            'history.openloop.{}-{:05d}.nc'.format(
            EnKF_start_time.strftime('%Y-%m-%d'),
            EnKF_start_time.hour*3600+EnKF_start_time.second))
postprocess_hist_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_postprocess_basedir'],
            'history',
            'history.openloop.{}-{:05d}.nc'.format(
            EnKF_start_time.strftime('%Y-%m-%d'),
            EnKF_start_time.hour*3600+EnKF_start_time.second))
truth_hist_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['EnKF']['truth_hist_nc'])

# For innovation map
ens_hist_dir = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_EnKF_basedir'],
            'history',
            'EnKF_ensemble_concat')
#ens_hist_dir = '/raid2/ymao/data_assim/output/EnKF/ArkRed.test/history/EnKF_ensemble_concat/sm1'  # filenames: ens<i>.nc
#ens_sm1_varname = 'sm1'  # sm1 varname in the netCDF files for each ensemble under ens_hist_dir
#ens_mean_hist_nc = '/raid2/ymao/data_assim/output/EnKF/ArkRed.test/history/EnKF_ensemble_concat/sm1/ensemble_mean_sm1.nc'  # Ensemble mean history file after EnKF step
#ens_mean_sm1_varname = 'ensemble_mean_sm1'  # sm1 varname in the netCDF file ens_mean_hist_nc
meas_nc = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['EnKF']['meas_nc'])
meas_sm1_varname = cfg['EnKF']['meas_var_name']
R = cfg['EnKF']['R']  # Measurement error variance
N = cfg['EnKF']['N']  # Number of ensemble members

# Output
output_dir = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_postprocess_basedir'],
            'plots')


# ======================================================== #
# Load and process data
# ======================================================== #
# --- Load data --- #
ds_openloop = xr.open_dataset(openloop_hist_nc)
print(postprocess_hist_nc)
exit()
ds_postprocess = xr.open_dataset(postprocess_hist_nc)
ds_truth = xr.open_dataset(truth_hist_nc)
ds_meas = xr.open_dataset(meas_nc)
# Load ensemble data
list_ds_ens = []
for i in range(N):
    list_ds_ens.append(xr.open_dataset(os.path.join(
                ens_hist_dir,
                'history.ens{}.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                        i+1,
                        EnKF_start_time.strftime('%Y%m%d'),
                        EnKF_start_time.hour*3600+EnKF_start_time.second,
                        EnKF_end_time.strftime('%Y%m%d'),
                        EnKF_end_time.hour*3600+EnKF_end_time.second))))

# --- Cut data to the same time period --- #
start_time = EnKF_start_time
end_time = EnKF_end_time
ds_truth = ds_truth.sel(time=slice(start_time, end_time))
ds_openloop = ds_openloop.sel(time=slice(start_time, end_time))
ds_meas = ds_meas.sel(time=slice(start_time, end_time))
for i, ds in enumerate(list_ds_ens):
    ds = ds.sel(time=slice(start_time, end_time))
    list_ds_ens[i] = ds









