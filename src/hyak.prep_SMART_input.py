
''' This script prepares direct input files for SMART run, including:
        - Input data .mat file (including all precipitation datasets and soil moisture datasets)
        - Matlab running bash script

    Usage:
        $ python prep_SMART_input.py config_file
'''

import xarray as xr
import sys
import os
import datetime as dt
import pandas as pd
import numpy as np
from scipy.io import savemat
import numbers

from tonic.io import read_configobj

from da_utils import (load_nc_and_concat_var_years, setup_output_dirs,
                      da_3D_to_2D_for_SMART)


# ============================================================ #
# Process command line arguments
# Read config file
# ============================================================ #
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Process some input variables
# ============================================================ #
start_time = dt.datetime.strptime(cfg['SMART_RUN']['start_time'], "%Y-%m-%d-%H")
end_time = dt.datetime.strptime(cfg['SMART_RUN']['end_time'], "%Y-%m-%d-%H")
start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Load input datasets
# ============================================================ #

dict_da = {}

# --- Original prec (to be corrected) --- #
da_prec_orig = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_orig_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_orig': cfg['PREC']['prec_orig_varname']})\
                  ['prec_orig'].sel(time=slice(start_time, end_time))
# put in dict
dict_da['prec_orig'] = da_prec_orig

# --- Independent prec (for lambda tuning) --- #
da_prec_indep = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_indep_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_indep': cfg['PREC']['prec_indep_varname']})\
                 ['prec_indep'].sel(time=slice(start_time, end_time))
# put in dict
dict_da['prec_for_tuning_lambda'] = da_prec_indep

# --- "True" prec --- #
da_prec_true = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_true_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_true': cfg['PREC']['prec_true_varname']})\
               ['prec_true'].sel(time=slice(start_time, end_time))
# put in dict
dict_da['prec_true'] = da_prec_true

# --- Soil moisture --- #
# If ascending and descending products are in separate files, directly load
if cfg['SM']['sep_am_pm']:
    # Ascending
    if cfg['SM']['sm_ascend_nc'] is not None:
        da_sm_ascend = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                                    cfg['SM']['sm_ascend_nc']))\
                       [cfg['SM']['sm_ascend_varname']]
        # put in dict
        dict_da['sm_ascend'] = da_sm_ascend
    else:
        da_sm_ascend = dict_da['prec_orig'].copy(deep=True)
        da_sm_ascend
    # Descending
    if cfg['SM']['sm_descend_nc'] is not None:
        da_sm_descend = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                                    cfg['SM']['sm_descend_nc']))\
                        [cfg['SM']['sm_descend_varname']]
        # put in dict
        dict_da['sm_descend'] = da_sm_descend
# If ascending and descending products are in the same file, load and then
# identify and separate ascending and descending
else:
    # Load
    da_sm = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'], cfg['SM']['sm_nc']))\
        [cfg['SM']['sm_varname']]
    # Extract ascending and descending data
    da_sm_ascend = da_sm.sel(
        time=pd.to_datetime(da_sm['time'].values).hour == cfg['SM']['ascend_hour'])
    da_sm_descend = da_sm.sel(
        time=pd.to_datetime(da_sm['time'].values).hour == cfg['SM']['descend_hour'])
    # Check whether the separated data has the same total lenght with the original data
    if len(da_sm_ascend['time']) + len(da_sm_descend['time']) != len(da_sm['time']):
        raise ValueError('Separated ascending and descending SM data does not add'
                         'up to the total lenght of the original SM data!')
    # Put in dict
    dict_da['sm_ascend'] = da_sm_ascend
    dict_da['sm_descend'] = da_sm_descend


# ============================================================ #
# If time_step != 24:
# 1) shift SM measurements to one-timestep earlier
# since SMART assumes the SM on the corresponding timesetp to be period-end;
# 1) Also make the SM data to have the same timestep as precipitation
# by filling in NAN for unobserved timesteps
# ============================================================ #
if cfg['SMART_RUN']['time_step'] != 24:
    for var in ['sm_ascend', 'sm_descend']:
        da_new = dict_da['prec_orig'].copy(deep=True)
        da_new[:] = np.nan  # [time, lat, lon]
        # If not missing
        if var in dict_da.keys():
            # Shift time
            times_new = [pd.to_datetime(t) - pd.DateOffset(hours=cfg['SMART_RUN']['time_step'])
                         for t in dict_da[var]['time'].values]
            # Fill in nans for unobserved timesteps
            da_new.loc[times_new, :, :] = dict_da[var].values
        # If missing, keep all NAN values
        # Put into dict
        dict_da[var] = da_new


# ============================================================ #
# If time_step = 24, aggregate all data to daily;
# ============================================================ #
if cfg['SMART_RUN']['time_step'] == 24:
    dict_da_daily = {}

    # --- Aggregate prec variables to daily (orig. units: mm/step) --- #
    for var, da in dict_da.items():
        # If precipitation variable
        if var == 'prec_orig' or var == 'prec_for_tuning_lambda' or\
        var == 'prec_true':
            # Sum daily preciptation
            da_daily = da.resample(time='1D').sum(dim='time')
            # Put into dict
            dict_da_daily[var] = da_daily

    # --- Average and process soil moisture variables to daily --- #
    for var in ['sm_ascend', 'sm_descend']:
        # Copy the shape of the DataArray from a prec variable
        da_sm = dict_da_daily['prec_orig'].copy()
        da_sm[:] = np.nan
        # If not missing
        if var in dict_da.keys():
            # Average to daily soil moisture
            da = dict_da[var]
            da_daily = da.resample(time='1D').mean(dim='time')
            # Put into dict
            # NOTE: (assume the "daily-mean" SM represents the SM at 00:00 on the date)
            # (since SMART assumes the SM observation on the corresponding day with rainfall
            # to be at the end of the day, we need to shift the SM date forward by one day)
            da_daily = da_daily.shift(time=-1)
            da_sm.loc[da_daily['time'], :, :] = da_daily[:]
            dict_da_daily[var] = da_sm
        # If missing, put in NAN
        else:
            # Put into dict
            dict_da_daily[var] = da_sm


# ============================================================ #
# Convert data to dimension [npixel_active, ntimes]
# ============================================================ #
# Load in domain file
ds_domain = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['DOMAIN']['domain_file']))
da_mask = ds_domain['mask']

# Convert data to dimension [npixel_active, nday]
if cfg['SMART_RUN']['time_step'] == 24:
    dict_to_convert = dict_da_daily
else:
    dict_to_convert = dict_da
dict_array_active = da_3D_to_2D_for_SMART(dict_to_convert,
                                          da_mask,
                                          time_varname='time')


# ============================================================ #
# Make soil moisture uncertainty data
# ============================================================ #
# Copy the data for shape
sm_error = dict_array_active['sm_ascend'].copy()  # [npixel, ntime]
sm_error[:] = np.nan

# If input is a numerical number, assign spatial constant R values
if isinstance(cfg['SM']['R'], numbers.Number):
    std = np.sqrt(cfg['SM']['R'])
    sm_error[:] = std
# If input is an xr.Dataset
else:
    if cfg['SM']['R_vartype'] == 'R':
        da_R = xr.open_dataset(
            os.path.join(cfg['CONTROL']['root_dir'], cfg['SM']['R']))\
            [cfg['SM']['R_varname']]
        da_std = da_R.sqrt()
    elif cfg['SM']['R_vartype'] == 'std':
        da_std = xr.open_dataset(
            os.path.join(cfg['CONTROL']['root_dir'], cfg['SM']['R']))\
            [cfg['SM']['R_varname']]
    # Convert 2D field to 3D temporally-constant field
    ntime = sm_error.shape[1]
    da_std_3D = xr.DataArray(
        np.zeros([ntime, len(da_std['lat']), len(da_std['lon'])]),
        coords=[range(ntime), da_std['lat'], da_std['lon']],
        dims=['time', 'lat', 'lon'])
    da_std_3D[:] = da_std
    # Convert 3D da to [npixel, ntime]
    sm_error = da_3D_to_2D_for_SMART(
        {'sm_error': da_std_3D},
        da_mask,
        time_varname='time')['sm_error']
    
# Put in final dictionary
dict_array_active['sm_error'] = sm_error


# ============================================================ #
# Save datasets to .mat file
# ============================================================ #
# Set up output subdir for all outputs from this preparation script
out_dir = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['OUTPUT']['output_basedir']),
                            mkdirs=['prep_SMART'])['prep_SMART']

# Save datasets to .mat file
savemat(os.path.join(out_dir, 'SMART_input.mat'), dict_array_active)


# ============================================================ #
# Prepare Matlab running bash script
# ============================================================ #

# Set up SMART run output subdir
smart_run_outdir = setup_output_dirs(
                        os.path.join(cfg['CONTROL']['root_dir'],
                                     cfg['OUTPUT']['output_basedir']),
                        mkdirs=['run_SMART'])['run_SMART']

# Open file for writing
with open(os.path.join(out_dir, 'run.m'), 'w') as f:
    # Header lines
    f.write('clear all; close all; clc;\n\n')
    # cd to SMART matlab code directory
    f.write('cd {}\n\n'.format(os.path.join(
                                    cfg['CONTROL']['root_dir'],
                                    cfg['SMART_MATLAB']['matlab_dir'])))
    # Call SMART function with all options
    f.write('SMART(\'input_dataset\', \'{}\', ...\n'.format(
                        os.path.join(out_dir,
                                     'SMART_input.mat')))
    f.write('    \'output_dir\', \'{}\', ...\n'.format(smart_run_outdir))
    f.write('    \'start_time\', \'{}\', ...\n'.format(start_time.strftime('%Y-%m-%d %H:%M')))
    f.write('    \'end_time\', \'{}\', ...\n'.format(end_time.strftime('%Y-%m-%d %H:%M')))
    f.write('    \'time_step\', \'{}\', ...\n'.format(cfg['SMART_RUN']['time_step']))
    f.write('    \'filter_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['filter_flag']))
    f.write('    \'transform_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['transform_flag']))
    f.write('    \'API_model_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['API_model_flag']))
    f.write('    \'lambda_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['lambda_flag']))
    f.write('    \'NUMEN\', \'{}\', ...\n'.format(cfg['SMART_RUN']['NUMEN']))
    f.write('    \'Q_fixed\', \'{}\', ...\n'.format(cfg['SMART_RUN']['Q_fixed']))
    f.write('    \'P_inflation\', \'{}\', ...\n'.format(cfg['SMART_RUN']['P_inflation']))
    f.write('    \'upper_bound_API\', \'{}\', ...\n'.format(cfg['SMART_RUN']['upper_bound_API']))
    f.write('    \'logn_var\', \'{}\', ...\n'.format(cfg['SMART_RUN']['logn_var']))
    f.write('    \'phi\', \'{}\', ...\n'.format(cfg['SMART_RUN']['phi']))
    f.write('    \'slope_parameter_API\', \'{}\', ...\n'.format(cfg['SMART_RUN']['slope_parameter_API']))
    f.write('    \'location_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['location_flag']))
    f.write('    \'window_size\', \'{}\', ...\n'.format(cfg['SMART_RUN']['window_size']))
    f.write('    \'API_mean\', \'{}\', ...\n'.format(cfg['SMART_RUN']['API_mean']))
    f.write('    \'bb\', \'{}\', ...\n'.format(cfg['SMART_RUN']['bb']))
    f.write('    \'API_range\', \'{}\', ...\n'.format(cfg['SMART_RUN']['API_range']))
    f.write('    \'if_rescale\', \'{}\', ...\n'.format(cfg['SMART_RUN']['if_rescale']))
    f.write('    \'lambda_tuning_target\', \'{}\', ...\n'.format(cfg['SMART_RUN']['lambda_tuning_target']))
    f.write('    \'correct_magnitude_only\', \'{}\', ...\n'.format(cfg['SMART_RUN']['correct_magnitude_only']))
    f.write('    \'correct_magnitude_only_threshold\', \'{}\', ...\n'.format(cfg['SMART_RUN']['correct_magnitude_only_threshold']))
    f.write('    \'sep_sm_orbit\', \'{}\')\n'.format(cfg['SMART_RUN']['sep_sm_orbit']))


