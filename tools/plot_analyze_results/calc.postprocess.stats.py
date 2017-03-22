
# Usage:
#   python analyze.py <config_file> <debug>

import matplotlib
matplotlib.use('Agg')
import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import sys
import multiprocessing as mp
from collections import OrderedDict

from tonic.io import read_config, read_configobj


def load_nc_file_year(nc_file, N):
    ''' Loads in nc files for one year, but all ensemble members.

    Parameters
    ----------
    nc_file: <str>
        netCDF file to load, with {} to be substituted by ensemble index
    N: <int>
        Ensemble size

    Returns
    ----------
    ds_all_ens: <xr.Dataset>
        Dataset of all ensemble members
    '''
    
    list_ds = []
    for i in range(N):
        # Load data
        fname = nc_file.format(i+1)
        ds = xr.open_dataset(fname)
        list_ds.append(ds)
    # Concat all years
    ds_all_ens = xr.concat(list_ds, dim='N')

    return ds_all_ens


def setup_output_dirs(out_basedir, mkdirs=['results', 'state',
                                            'logs', 'plots']):
    ''' This function creates output directories.

    Parameters
    ----------
    out_basedir: <str>
        Output base directory for all output files
    mkdirs: <list>
        A list of subdirectories to make

    Require
    ----------
    os
    OrderedDict

    Returns
    ----------
    dirs: <OrderedDict>
        A dictionary of subdirectories

    '''

    dirs = OrderedDict()
    for d in mkdirs:
        dirs[d] = os.path.join(out_basedir, d)

    for dirname in dirs.values():
        os.makedirs(dirname, exist_ok=True)

    return dirs


def to_netcdf_history_file_compress(ds_hist, out_nc):
    ''' This function saves a VIC-history-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_hist: <xr.Dataset>
        History dataset to save
    out_nc: <str>
        Path of output netCDF file
    '''

    dict_encode = {}
    for var in ds_hist.data_vars:
        # skip variables not starting with "OUT_"
        if var.split('_')[0] != 'OUT':
            continue
        # determine chunksizes
        chunksizes = []
        for i, dim in enumerate(ds_hist[var].dims):
            if dim == 'time':  # for time dimension, chunksize = 1
                chunksizes.append(1)
            else:
                chunksizes.append(len(ds_hist[dim]))
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1,
                            'chunksizes': chunksizes}
    ds_hist.to_netcdf(out_nc,
                      format='NETCDF4',
                      encoding=dict_encode)


def calc_ens_stats(nc_file, N, output_ens_mean_subdir,
                   output_ens_median_subdir, year):
    ''' Calculates cross-ensemble statistics.

    Parameters
    ----------
    nc_file: <str>
        netCDF file to load, with {} to be substituted by ensemble index
    N: <int>
        Ensemble size
    output_ens_mean_subdir: <str>
        Output directory for mean results
    output_ens_median_subdir: <str>
        Output directory for median results
    year: <int>
        Year of file
    '''
    ds_all_ens = load_nc_file_year(nc_file, N)
    # Calculate and save ensemble mean
    ds_ens_mean = ds_all_ens.mean(dim='N')
    to_netcdf_history_file_compress(
            ds_ens_mean,
            os.path.join(output_ens_mean_subdir,
                         'ens_mean.{}.nc'.format(year)))
    # Calculate and save ensemble median
    ds_ens_median = ds_all_ens.median(dim='N')
    to_netcdf_history_file_compress(
            ds_ens_median,
            os.path.join(output_ens_median_subdir,
                         'ens_median.{}.nc'.format(year)))


# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.argv[1])

# Number of processors for parallelizing each year
nproc = int(sys.argv[2])

# ========================================================== #
# Parameter setting
# ========================================================== #

if 'LINEAR_MODEL' in cfg:
    linear_model = True
else:
    linear_model = False

# --- Input directory and files --- #
# Post-process results
post_result_basedir = cfg['POSTPROCESS']['post_result_basedir']

# gen_synthetic results
gen_synth_basedir = cfg['EnKF']['gen_synth_basedir']
truth_nc_filename = cfg['EnKF']['truth_nc_filename']
synth_meas_nc_filename = cfg['EnKF']['synth_meas_nc_filename']

# openloop
openloop_nc = cfg['EnKF']['openloop_nc']

# initial time
init_time = pd.to_datetime(cfg['EnKF']['init_time'])

# VIC global file template (for extracting param file and snow_band)
vic_global_txt = cfg['EnKF']['vic_global_txt']

# Forcings (for all basepaths, 'YYYY.nc' will be appended)
orig_force_basepath = cfg['EnKF']['orig_force_basepath']
truth_force_basepath = cfg['EnKF']['truth_force_basepath']
# ens_force_basedir/ens_<i>/force.<YYYY>.nc, where <i> = 1, 2, ..., N
ens_force_basedir = cfg['EnKF']['ens_force_basedir']

# VIC parameter netCDF file
vic_param_nc = cfg['EnKF']['vic_param_nc']
    
# --- Measurement times --- #
times = pd.date_range(cfg['EnKF']['meas_start_time'],
                      cfg['EnKF']['meas_end_time'],
                      freq=cfg['EnKF']['freq'])
ntime = len(times)

# --- Plot time period --- #
plot_start_time = pd.to_datetime(cfg['EnKF']['plot_start_time'])
plot_end_time = pd.to_datetime(cfg['EnKF']['plot_end_time'])
start_year = plot_start_time.year
end_year = plot_end_time.year

# --- others --- #
N = cfg['EnKF']['N']  # number of ensemble members
ens = cfg['EnKF']['ens']  # index of ensemble member to plot for debugging plots

# --- Output --- #
output_dir = cfg['POSTPROCESS']['output_post_dir']

# ========================================================== #
# Setup output data dir
# ========================================================== #
output_data_dir = setup_output_dirs(
                    output_dir,
                    mkdirs=['data'])['data']
output_ens_mean_subdir = setup_output_dirs(
                    output_data_dir,
                    mkdirs=['ens_mean'])['ens_mean']
output_ens_median_subdir = setup_output_dirs(
                    output_data_dir,
                    mkdirs=['ens_median'])['ens_median']

# ========================================================== #
# Load data; calculate ensemble mean and median
# ========================================================== #
# --- Loop over each year --- #
if nproc == 1:
    for year in range(start_year, end_year+1):
        print('\t\t{}'.format(year))
        # Load in results for all ensemble members
        nc_file = os.path.join(
                post_result_basedir,
                'history',
                'ens{}',
                'history.concat.{}.nc'.format(year))
        calc_ens_stats(nc_file, N, output_ens_mean_subdir,
                       output_ens_median_subdir, year)
elif nproc > 1:
    # --- Set up multiprocessing --- #
    pool = mp.Pool(processes=nproc)
    # --- Loop over each year --- #
    for year in range(start_year, end_year+1):
        print('\t\t{}'.format(year))
        # Load in results for all ensemble members
        nc_file = os.path.join(
                post_result_basedir,
                'history',
                'ens{}',
                'history.concat.{}.nc'.format(year))
        pool.apply_async(calc_ens_stats,
                         (nc_file, N, output_ens_mean_subdir,
                          output_ens_median_subdir, year))
    # --- Finish multiprocessing --- #
    pool.close()
    pool.join()

# --- Concat all years --- #
# --- Mean --- #
print('Concatenating mean of all years...')
list_ds = []
list_files_to_delete = []
for year in range(start_year, end_year+1):
    fname = os.path.join(output_ens_mean_subdir,
                         'ens_mean.{}.nc'.format(year))
    ds = xr.open_dataset(fname)
    list_ds.append(ds)
    list_files_to_delete.append(fname)
ds_concat = xr.concat(list_ds, dim='time')
to_netcdf_history_file_compress(
        ds_concat,
        os.path.join(output_ens_mean_subdir,
                     'ens_mean.concat.{}_{}.nc'.format(start_year, end_year)))
# Clean up individual years of files
for f in list_files_to_delete:
    os.remove(f)

# --- Median --- #
print('Concatenating median of all years...')
list_ds = []
list_files_to_delete = []
for year in range(start_year, end_year+1):
    fname = os.path.join(output_ens_median_subdir,
                         'ens_median.{}.nc'.format(year))
    ds = xr.open_dataset(fname)
    list_ds.append(ds)
    list_files_to_delete.append(fname)
ds_concat = xr.concat(list_ds, dim='time')
to_netcdf_history_file_compress(
        ds_concat,
        os.path.join(output_ens_median_subdir,
                     'ens_median.concat.{}_{}.nc'.format(start_year, end_year)))
# Clean up individual years of files
for f in list_files_to_delete:
    os.remove(f)


