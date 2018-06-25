
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

from analysis_utils import (setup_output_dirs, to_netcdf_state_file_compress,
                            determine_tile_frac)


def concat_updated_states(i, EnKF_result_basedir, meas_times,
                          output_concat_states_dir, global_template):
    ''' A wrap function that concatenates EnKF updated SM states.

    Parameters
    ----------
    i: <int>
        Index of ensemble member (starting from 0)
    EnKF_result_basedir: <str>
        EnKF output result base directory
    meas_times: <list or pandas.tseries.index.DatetimeIndex>
        Measurement time points; the same as state updating time points
    output_concat_states_dir: <str>
        Directory for outputing concatenated SM state files
    global_template: <str>
        VIC global file path

    Returns
    ----------
    da_state_all_times: <xr.DataArray>
        Concatenated SM states for this ensemble member
    '''

    # --- Load states at measurement times --- #
    list_da_state = []
    list_da_swe = []
    for t in meas_times:
        state_nc = os.path.join(
            EnKF_result_basedir,
            'states',
            'updated.{}_{:05d}'.format(
                t.strftime('%Y%m%d'),
                t.hour*3600+t.second),
            'state.ens{}.nc'.format(i+1))
        da_state = xr.open_dataset(state_nc)['STATE_SOIL_MOISTURE']
        da_swe = xr.open_dataset(state_nc)['STATE_SNOW_WATER_EQUIVALENT']
        list_da_state.append(da_state)
        list_da_swe.append(da_swe)
    # --- Concatenate states of all time together --- #
    da_state_all_times = xr.concat(list_da_state, dim='time')
    da_state_all_times['time'] = meas_times
    da_swe_all_times = xr.concat(list_da_swe, dim='time')
    da_swe_all_times['time'] = meas_times
#    # --- Save concatenated states to netCDF file --- #
#    ds_state_all_times = xr.Dataset(
#        {'STATE_SOIL_MOISTURE': da_state_all_times,
#         'STATE_SNOW_WATER_EQUIVALENT': da_swe_all_times})
#    out_nc = os.path.join(
#        output_concat_states_dir,
#        'updated_state.{}_{}.ens{}.nc'.format(
#            meas_times[0].strftime('%Y%m%d'),
#            meas_times[-1].strftime('%Y%m%d'),
#            i+1))
#    to_netcdf_state_file_compress(
#        ds_state_all_times, out_nc)
    # Calculate and save cell-average states to netCDF file
    da_tile_frac = determine_tile_frac(global_template)
    da_state_cellAvg = (da_state_all_times * da_tile_frac).sum(
        dim='veg_class').sum(dim='snow_band')  # [time, nlayer, lat, lon]
    da_swe_cellAvg = (da_swe_all_times * da_tile_frac).sum(
        dim='veg_class').sum(dim='snow_band')  # [time, lat, lon]
    ds_state_cellAvg = xr.Dataset({'SOIL_MOISTURE': da_state_cellAvg,
                                  'SWE': da_swe_cellAvg})
    out_nc = os.path.join(
        output_concat_states_dir,
        'updated_state_cellAvg.{}_{}.ens{}.nc'.format(
            meas_times[0].strftime('%Y%m%d'),
            meas_times[-1].strftime('%Y%m%d'),
            i+1))
    to_netcdf_state_file_compress(
        ds_state_cellAvg, out_nc)


# ========================================================== #
# Command line arguments
# ========================================================== #
# --- Load in config file --- #
cfg = read_configobj(sys.argv[1])

# --- Number of processors --- #
nproc = int(sys.argv[2])

# --- Ensemble index --- #
ens = int(sys.argv[3])

# ========================================================== #
# Parameter setting
# ========================================================== #

# --- Input directory and files --- #
EnKF_result_basedir = cfg['EnKF']['EnKF_result_basedir']

# VIC global file template (for extracting param file and snow_band)
vic_global_txt = cfg['EnKF']['vic_global_txt']

# VIC parameter netCDF file
vic_param_nc = cfg['EnKF']['vic_param_nc']
    
# --- Measurement times --- #
meas_times = pd.date_range(
    cfg['EnKF']['meas_start_time'],
    cfg['EnKF']['meas_end_time'],
    freq=cfg['EnKF']['freq'])

# --- others --- #
N = cfg['EnKF']['N']  # number of ensemble members

# ========================================================== #
# Setup output data dir
# ========================================================== #
output_concat_states_dir = setup_output_dirs(
    os.path.join(EnKF_result_basedir, 'states'),
    mkdirs=['updated_concat'])['updated_concat']

# ========================================================== #
# Concat SM & SWE states for each ensemble member
# ========================================================== #
print('Concatenating updated SM states for each ensemble member...')

# --- If nproc == 1, do a regular ensemble loop --- #
if nproc == 1:
#    for i in range(N):
    for i in range(ens-1, ens):
        print('Ensemble {}'.format(i))
        concat_updated_states(
            i, EnKF_result_basedir, meas_times,
            output_concat_states_dir, vic_global_txt)
# --- If nproc > 1, use multiprocessing --- #
elif nproc > 1:
    # --- Set up multiprocessing --- #
    pool = mp.Pool(processes=nproc)
    # --- Loop over each ensemble member --- #
    for i in range(N):
        print('Ensemble {}'.format(i))
        pool.apply_async(concat_updated_states,
                         (i, EnKF_result_basedir, meas_times,
                         output_concat_states_dir, vic_global_txt))
    # --- Finish multiprocessing --- #
    pool.close()
    pool.join()


## ========================================================== #
## Calculate ensemble-mean updated SM & SWE states
## ========================================================== #
## --- Calculate ensemble-mean updated SM states --- #
#print('Calculating ensemble-mean updated SM states...')
#da_all_ens = xr.concat(list_da_concat, dim='N')
#da_ens_mean = da_all_ens.mean(dim='N')
## Save ensemble-mean updated SM states to netCDF file
#ds_ens_mean = xr.Dataset(
#    {'STATE_SOIL_MOISTURE': da_ens_mean})
#out_nc = os.path.join(
#    output_concat_states_dir,
#    'updated_state.{}_{}.ens_mean.nc'.format(
#        meas_times[0].strftime('%Y%m%d'),
#        meas_times[-1].strftime('%Y%m%d')))
#to_netcdf_state_file_compress(ds_ens_mean, out_nc)
#
## --- Calculate ensemble-mean cell-avg updated SM states --- #
#print('Calculating ensemble-mean cell-avg updated SM states...')
#da_tile_frac = determine_tile_frac(vic_global_txt)
#da_state_cellAvg = (da_ens_mean * da_tile_frac).sum(
#    dim='veg_class').sum(dim='snow_band')  # [time, nlayer, lat, lon]
#ds_state_cellAvg = xr.Dataset({'SOIL_MOISTURE': da_state_cellAvg})
#out_nc = os.path.join(
#    output_concat_states_dir,
#    'updated_state_cellAvg.{}_{}.ens_mean.nc'.format(
#        meas_times[0].strftime('%Y%m%d'),
#        meas_times[-1].strftime('%Y%m%d')))
#to_netcdf_state_file_compress(
#    ds_state_cellAvg, out_nc)
#
#
#
#
