
''' This script post-processes SMART output - it rescales the original
    pricipitation data based on the SMART-output window-sum corrected prec.

    Currently, do not correct (i.e., use original un-corrected prec for): 
        1) the first window (SMART always outputs zero for the first window)
        2) the timesteps after the last complete window

    Usage:
        $ python postprocess_SMART.py <config_file_SMART>
'''

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
import multiprocessing as mp

from tonic.io import read_configobj

from da_utils import (load_nc_and_concat_var_years, setup_output_dirs,
                      da_2D_to_3D_from_SMART, da_3D_to_2D_for_SMART,
                      correct_prec_from_SMART, rmse, to_netcdf_forcing_file_compress,
                      rescale_and_save_SMART_prec, save_SMART_prec)


# ============================================================ #
# Process command line arguments
# Read config file
# ============================================================ #
cfg = read_configobj(sys.argv[1])

nproc = int(sys.argv[2])


# ============================================================ #
# Process some input variables
# ============================================================ #
start_time = pd.to_datetime(cfg['SMART_RUN']['start_time'])
end_time = pd.to_datetime(cfg['SMART_RUN']['end_time'])
start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Load original and corrected (window-average) prec data
# ============================================================ #
print('Loading data...')

# --- Load original prec (to be corrected) --- #
da_prec_orig = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_orig_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_orig': cfg['PREC']['prec_orig_varname']})\
                  ['prec_orig'].sel(time=slice(start_time, end_time))

# --- Load corrected window-averaged prec --- #
run_SMART_outfile = os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['OUTPUT']['output_basedir'],
                                 'run_SMART',
                                 'SMART_corrected_rainfall.mat')
run_SMART_prec_corr = loadmat(run_SMART_outfile)['RAIN_SMART_SMOS']  # [nwindow, npixel]

# --- Load corrected window-averaged prec ensemble (if ensemble SMART) --- #
filter_flag = cfg['SMART_RUN']['filter_flag']
if filter_flag == 2 or filter_flag == 6:
    list_run_SMART_prec_corr_ens = []
    for i in range(cfg['SMART_RUN']['NUMEN']):
        run_SMART_ens_outfile = os.path.join(
            cfg['CONTROL']['root_dir'],
            cfg['OUTPUT']['output_basedir'],
            'run_SMART',
            'SMART_corrected_rainfall.ens{}.mat'.format(i+1))
        run_SMART_prec_corr_ens = loadmat(
            run_SMART_ens_outfile)['RAIN_CORRECTED']['ens{}'.format(i+1)][0][0].squeeze()  # [nwindow, npixel]
        # If npixel = 1, Matlab automatically squeezes that dimension. Here we readd this dimension
        if len(run_SMART_prec_corr_ens.shape) == 1:
            run_SMART_prec_corr_ens = run_SMART_prec_corr_ens.reshape(
                [run_SMART_prec_corr_ens.shape[0], 1])
        # Put in list
        list_run_SMART_prec_corr_ens.append(run_SMART_prec_corr_ens)

# Load in domain file
ds_domain = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['DOMAIN']['domain_file']))
da_mask = ds_domain['mask']


# ============================================================ #
# Process and save window-averaged SMART-corrected prec
# ============================================================ #
print('Processing and saving window-averaged precip...')
# --- Precess prec data to da --- #
# Preprocess SMART output prec
dict_prec_SMART_window = {'prec_corr_window': run_SMART_prec_corr}
nwindow = run_SMART_prec_corr.shape[0]
da_prec_corr_window = da_2D_to_3D_from_SMART(
                            dict_array_2D=dict_prec_SMART_window,
                            da_mask=da_mask,
                            out_time_varname='window',
                            out_time_coord=range(nwindow))['prec_corr_window']
# Process SMART prec ensemble
if filter_flag == 2 or filter_flag == 6:
    dict_prec_SMART_window_ens = {}
    for i in range(cfg['SMART_RUN']['NUMEN']):
        dict_prec_SMART_window_ens[i+1] = list_run_SMART_prec_corr_ens[i]
    dict_da_prec_corr_window_ens = da_2D_to_3D_from_SMART(
            dict_array_2D=dict_prec_SMART_window_ens,
            da_mask=da_mask,
            out_time_varname='window',
            out_time_coord=range(nwindow))

# --- Save window-averaged SMART-corrected prec --- #
# Set up output subdir
out_dir = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['OUTPUT']['output_basedir']),
                            mkdirs=['post_SMART'])['post_SMART']
# Save mean window-averaged prec
ds_prec_corrected_window = xr.Dataset({'prec_corrected_window': da_prec_corr_window})
to_netcdf_forcing_file_compress(
    ds_force=ds_prec_corrected_window,
    out_nc=os.path.join(out_dir, 'prec_corrected_window.nc'),
    time_dim='window')


# ============================================================ #
# Rescale orig. prec at orig. timestep based on SMART outputs
# and save to netCDF file;
# This rescaling step is only needed if window_size > 1
# ============================================================ #
print('Rescaling original prec. and saving to netCDF...')

# --- If window_size == 1, skip rescaling and directly save
print('window_size = 1, skip rescaling')
if cfg['SMART_RUN']['window_size'] == 1:
    # Save deterministic SMART-corrected precip
    save_SMART_prec(start_time, end_time, cfg['SMART_RUN']['time_step'],
                    da_prec_corr_window, out_dir, 'prec_corrected.')
    # Save ensemble corrected precip, if applicable
    if filter_flag == 2 or filter_flag == 6:
        # Loop over all ensemble members
        # --- If nproc == 1, do a regular ensemble loop --- #
        if nproc == 1:
            for i in range(cfg['SMART_RUN']['NUMEN']):
                print('\tEnsemble {}'.format(i))
                save_SMART_prec(start_time, end_time, cfg['SMART_RUN']['time_step'],
                                dict_da_prec_corr_window_ens[i+1],
                                out_dir, 'prec_corrected.ens{}.'.format(i+1))
        # --- If nproc > 1, use multiprocessing --- #
        elif nproc > 1:
            # --- Set up multiprocessing --- #
            pool = mp.Pool(processes=nproc)
            # --- Loop over each ensemble member --- #
            for i in range(cfg['SMART_RUN']['NUMEN']):
                print('\tEnsemble {}'.format(i))
                pool.apply_async(save_SMART_prec,
                                 (start_time, end_time, cfg['SMART_RUN']['time_step'],
                                  dict_da_prec_corr_window_ens[i+1],
                                  out_dir, 'prec_corrected.ens{}.'.format(i+1)))
            # --- Finish multiprocessing --- #
            pool.close()
            pool.join()


# --- If window_size > 1, rescale and save --- #
else:
    # --- Rescale SMART-corrected precip --- #
    da_prec_corrected = rescale_and_save_SMART_prec(
        da_prec_orig, cfg['SMART_RUN']['window_size'],
        cfg['SMART_RUN']['time_step'],
        da_prec_corr_window, start_time,
        out_dir, 'prec_corrected.')
    
    # --- If ensemble SMART, rescale each ensemble member --- #
    if filter_flag == 2 or filter_flag == 6:
        # --- If nproc == 1, do a regular ensemble loop --- #
        if nproc == 1:
            for i in range(cfg['SMART_RUN']['NUMEN']):
                print('Rescale and save ensemble {}...'.format(i+1))
                rescale_and_save_SMART_prec(
                    da_prec_orig, cfg['SMART_RUN']['window_size'],
                    cfg['SMART_RUN']['time_step'],
                    dict_da_prec_corr_window_ens[i+1],
                    start_time,
                    out_dir, 'prec_corrected.ens{}.'.format(i+1))
        # --- If nproc > 1, use multiprocessing --- #
        elif nproc > 1:
            # --- Set up multiprocessing --- #
            pool = mp.Pool(processes=nproc)
            # --- Loop over each ensemble member --- #
            for i in range(cfg['SMART_RUN']['NUMEN']):
                print('Rescale and save ensemble {}...'.format(i+1))
                pool.apply_async(rescale_and_save_SMART_prec,
                    (da_prec_orig, cfg['SMART_RUN']['window_size'],
                     cfg['SMART_RUN']['time_step'],
                     dict_da_prec_corr_window_ens[i+1],
                     start_time,
                     out_dir, 'prec_corrected.ens{}.'.format(i+1)))
            # --- Finish multiprocessing --- #
            pool.close()
            pool.join()
 
