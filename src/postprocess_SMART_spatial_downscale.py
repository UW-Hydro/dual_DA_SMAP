
# This script spatially downscales SMART-corrected, coarser-resolution rainfall (36km)
# to the original finer resolution, keeping the original sub-grid spatial pattern
# Right now, the script only handles not-split-grid weight file!

import numpy as np
import sys
import pandas as pd
import os
import xarray as xr
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import xesmf as xe
import multiprocessing as mp

from tonic.io import read_configobj

from da_utils import (load_nc_and_concat_var_years, setup_output_dirs,
                      da_2D_to_3D_from_SMART, da_3D_to_2D_for_SMART,
                      to_netcdf_forcing_file_compress,
                      regrid_spatial_prec_and_save)


# ============================================================ #
# Process command line arguments
# Read config file
# ============================================================ #
cfg = read_configobj(sys.argv[1])

nproc = int(sys.argv[2])


# ============================================================ #
# Check with SPATIAL_DOWNSCALE section is in the cfg file
# ============================================================ #
if 'SPATIAL_DOWNSCALE' in cfg:
    pass
else:
    raise ValueError('Must have [SPATIAL_DOWNSCALE] section in the cfg file to post-regrid'
                     'SMART-corrected rainfall field!')


# ============================================================ #
# Process some input variables
# ============================================================ #
start_time = pd.to_datetime(cfg['SMART_RUN']['start_time'])
end_time = pd.to_datetime(cfg['SMART_RUN']['end_time'])
start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Identify and set up subdirs
# ============================================================ #
out_post_dir = os.path.join(cfg['CONTROL']['root_dir'],
                            cfg['OUTPUT']['output_basedir'],
                            'post_SMART')
out_post_regridded_dir = setup_output_dirs(
    os.path.join(cfg['CONTROL']['root_dir'],
                 cfg['OUTPUT']['output_basedir']),
    mkdirs=['post_spatial_downscaled'])['post_spatial_downscaled']


# ============================================================ #
# Load original and corrected (postprocessed, original timestep) prec data
# ============================================================ #
print('Loading data...')

# --- Load original prec (original resolution to regrid to) --- #
da_prec_orig = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['SPATIAL_DOWNSCALE']['prec_orig_resolution_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_orig': cfg['SPATIAL_DOWNSCALE']['prec_orig_varname']})\
                  ['prec_orig'].sel(time=slice(start_time, end_time))

# --- Load corrected (postprocessed) prec --- #
# Ensemble-mean
da_prec_corrected_ensMean = load_nc_and_concat_var_years(
    basepath=os.path.join(out_post_dir, 'prec_corrected.'),
    start_year=start_year,
    end_year=end_year,
    dict_vars={'PREC': 'prec_corrected'})\
    ['PREC'].sel(time=slice(start_time, end_time))
# Ensemble members
filter_flag = cfg['SMART_RUN']['filter_flag']
if filter_flag == 2 or filter_flag == 6:
    list_da_prec_corrected_ens = []
    for i in range(cfg['SMART_RUN']['NUMEN']):
        da = load_nc_and_concat_var_years(
            basepath=os.path.join(out_post_dir, 'prec_corrected.ens{}.'.format(i+1)),
            start_year=start_year,
            end_year=end_year,
            dict_vars={'PREC': 'prec_corrected'})\
            ['PREC'].sel(time=slice(start_time, end_time))
        list_da_prec_corrected_ens.append(da)

# --- Load in domain file --- #
ds_domain = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['DOMAIN']['domain_file']))
da_mask = ds_domain['mask']


# --- Load weight file --- #
n_source = len(da_prec_orig['lat']) * len(da_prec_orig['lon'])
n_target = len(da_prec_corrected_ensMean['lat']) * len(da_prec_corrected_ensMean['lon'])
A = xe.frontend.read_weights(
    os.path.join(cfg['CONTROL']['root_dir'], cfg['SPATIAL_DOWNSCALE']['weight_nc']),
    n_source, n_target)
weight_array = A.toarray()  # [n_target, n_source]
# Check whether weight array has split source cells
if (weight_array>0).sum(axis=0).max() != 1:
    raise ValueError('The script only takes weight file that does not split'
                     'source grid cells!')


# ============================================================ #
# Regrid SMART-corrected rainfall to its original spatial resolution
# Keep the original finer-grid spatial pattern (separately for each timestep)
# ============================================================ #
print('Regridding...')
# --- Ensemble--mean --- #
print('\tEnsemble mean')
da_prec_corrected_regridded_ensMean = regrid_spatial_prec_and_save(
    os.path.join(cfg['CONTROL']['root_dir'], cfg['SPATIAL_DOWNSCALE']['weight_nc']),
    da_prec_orig,
    da_prec_corrected_ensMean,
    da_mask,
    out_post_regridded_dir,
    'prec_corrected.')

# --- Each ensemble member --- #
if filter_flag == 2 or filter_flag == 6:
    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(cfg['SMART_RUN']['NUMEN']):
            print('\tEnsemble {}'.format(i+1))
            da_prec_corrected_regridded_ensMean = regrid_spatial_prec_and_save(
                os.path.join(cfg['CONTROL']['root_dir'], cfg['SPATIAL_DOWNSCALE']['weight_nc']),
                da_prec_orig,
                list_da_prec_corrected_ens[i],
                da_mask,
                out_post_regridded_dir,
                'prec_corrected.ens{}.'.format(i+1))
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(cfg['SMART_RUN']['NUMEN']):
            print('\tEnsemble {}'.format(i+1))
            pool.apply_async(
                regrid_spatial_prec_and_save,
                (os.path.join(cfg['CONTROL']['root_dir'], cfg['SPATIAL_DOWNSCALE']['weight_nc']),
                 da_prec_orig,
                 list_da_prec_corrected_ens[i],
                 da_mask,
                 out_post_regridded_dir,
                 'prec_corrected.ens{}.'.format(i+1)))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()


