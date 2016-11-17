
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
start_date = dt.datetime.strptime(cfg['SMART_RUN']['start_date'], "%Y-%m-%d")
end_date = dt.datetime.strptime(cfg['SMART_RUN']['end_date'], "%Y-%m-%d")
start_year = start_date.year
end_year = end_date.year


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
                  ['prec_orig']
# put in dict
dict_da['prec_orig'] = da_prec_orig

# --- Independent prec (for lambda tuning) --- #
da_prec_indep = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_indep_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_indep': cfg['PREC']['prec_indep_varname']})\
                 ['prec_indep']
# put in dict
dict_da['prec_for_tuning_lambda'] = da_prec_indep

# --- "True" prec --- #
da_prec_true = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_true_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_true': cfg['PREC']['prec_true_varname']})\
               ['prec_true']
# put in dict
dict_da['prec_true'] = da_prec_true

# --- Soil moisture --- #
# Ascending
if cfg['SM']['sm_ascend_nc'] is not None:
    da_sm_ascend = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                                cfg['SM']['sm_ascend_nc']))\
                   [cfg['SM']['sm_ascend_varname']]
    # put in dict
    dict_da['sm_ascend'] = da_sm_ascend
# Descending
if cfg['SM']['sm_descend_nc'] is not None:
    da_sm_descend = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                                cfg['SM']['sm_descend_nc']))\
                    [cfg['SM']['sm_ascend_varname']]
    # put in dict
    dict_da['sm_descend'] = da_sm_descend


# ============================================================ #
# Aggregate all data to daily
# ============================================================ #

dict_da_daily = {}

# --- Aggregate prec variables to daily (orig. units: mm/step) --- #
for var, da in dict_da.items():
    # If precipitation variable
    if var == 'prec_orig' or var == 'prec_for_tuning_lambda' or\
    var == 'prec_true':
        # Sum daily preciptation
        da_daily = da.groupby('time.date').sum(dim='time')
        # Put into dict
        dict_da_daily[var] = da_daily

# --- Average and process soil moisture variables to daily --- #
for var in ['sm_ascend', 'sm_descend']:
    # If not missing
    if var in dict_da.keys():
        # Average to daily soil moisture
        da = dict_da[var]
        da_daily = da.groupby('time.date').mean(dim='time')
        # Put into dict
        dict_da_daily[var] = da_daily
    
    # If missing, put in NAN
    else:
        # Copy the shape of the DataArray from a prec variable
        da_daily = dict_da_daily['prec_orig'].copy()
        # Fill in all values with NAN
        da_daily[:] = np.nan
        # Put into dict
        dict_da_daily[var] = da_daily


# ============================================================ #
# Convert data to dimension [npixel_active, nday]
# ============================================================ #
# Load in domain file
ds_domain = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['DOMAIN']['domain_file']))
da_mask = ds_domain['mask']

# Convert data to dimension [npixel_active, nday]
dict_array_active = da_3D_to_2D_for_SMART(dict_da_daily,
                                          da_mask,
                                          time_varname='date')

# ============================================================ #
# Make soil moisture uncertainty data
# ============================================================ #
# Copy the data for shape
sm_error = dict_array_active['sm_ascend'].copy()
# Fill in constant value
sm_error[:, :] = cfg['SM']['sm_error']
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
    f.write('    \'output_dataset\', \'{}\', ...\n'.format(
                        os.path.join(smart_run_outdir,
                        'SMART_output.mat')))
    f.write('    \'start_date\', \'{}\', ...\n'.format(cfg['SMART_RUN']['start_date']))
    f.write('    \'end_date\', \'{}\', ...\n'.format(cfg['SMART_RUN']['end_date']))
    f.write('    \'filter_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['filter_flag']))
    f.write('    \'transform_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['transform_flag']))
    f.write('    \'API_model_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['API_model_flag']))
    f.write('    \'lambda_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['lambda_flag']))
    f.write('    \'NUMEN\', \'{}\', ...\n'.format(cfg['SMART_RUN']['NUMEN']))
    f.write('    \'Q_fixed\', \'{}\', ...\n'.format(cfg['SMART_RUN']['Q_fixed']))
    f.write('    \'P_inflation\', \'{}\', ...\n'.format(cfg['SMART_RUN']['P_inflation']))
    f.write('    \'upper_bound_API\', \'{}\', ...\n'.format(cfg['SMART_RUN']['upper_bound_API']))
    f.write('    \'logn_var\', \'{}\', ...\n'.format(cfg['SMART_RUN']['logn_var']))
    f.write('    \'slope_parameter_API\', \'{}\', ...\n'.format(cfg['SMART_RUN']['slope_parameter_API']))
    f.write('    \'location_flag\', \'{}\', ...\n'.format(cfg['SMART_RUN']['location_flag']))
    f.write('    \'window_size\', \'{}\', ...\n'.format(cfg['SMART_RUN']['window_size']))
    f.write('    \'API_mean\', \'{}\', ...\n'.format(cfg['SMART_RUN']['API_mean']))
    f.write('    \'bb\', \'{}\', ...\n'.format(cfg['SMART_RUN']['bb']))
    f.write('    \'API_range\', \'{}\')\n'.format(cfg['SMART_RUN']['API_range']))


