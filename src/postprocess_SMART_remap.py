
# This script regrid a SMART-postprocessed rainfall field to a different resolution

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

from da_utils import remap_con, setup_output_dirs, remap_and_save_smart_prec


# ======================================================= #
# Process command line argument
# ======================================================= #
cfg = read_configobj(sys.argv[1])

nproc = int(sys.argv[2])


# ============================================================ #
# Check whether REMAP section is in the cfg file
# ============================================================ #
if 'REMAP' in cfg:
    pass
else:
    raise ValueError('Must have [REMAP] section in the cfg file to post-regrid'
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
# Identify precipitation source dir
if cfg['REMAP']['prec_source'] == 'post_SMART_spatial_downscale':
    prec_input_dir = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_basedir'],
        'post_spatial_downscaled')
elif cfg['REMAP']['prec_source'] == 'post_SMART':
    prec_input_dir = os.path.join(
        cfg['CONTROL']['root_dir'],
        cfg['OUTPUT']['output_basedir'],
        'post_SMART')
else:
    raise ValueError('Unsupported option for prec_source!')

# Set up output dir    
out_remapped_dir = setup_output_dirs(
    os.path.join(cfg['CONTROL']['root_dir'],
                 cfg['OUTPUT']['output_basedir']),
    mkdirs=['post_final_remapped'])['post_final_remapped']


# ============================================================ #
# Load input precipitation fields, remap, and save
# ============================================================ #
print('Remapping...')
# --- Load domain files --- #
da_domain_target = xr.open_dataset(
    os.path.join(cfg['CONTROL']['root_dir'],
                 cfg['REMAP']['target_domain_nc']))['mask']
da_domain_source = xr.open_dataset(
    os.path.join(cfg['CONTROL']['root_dir'],
                 cfg['REMAP']['source_domain_nc']))['mask']

# --- Deterministic rainfall --- #
print('\tDetermnistic')
da_remapped = remap_and_save_smart_prec(
    os.path.join(prec_input_dir, 'prec_corrected.'),
    start_year, end_year,
    da_domain_target, out_remapped_dir, 'prec_corrected.',
    reuse_weight=False, da_domain_source=da_domain_source)

# --- Ensemble rainfall --- #
filter_flag = cfg['SMART_RUN']['filter_flag']
if filter_flag == 2 or filter_flag == 6:
    # --- If nproc == 1, do a regular ensemble loop --- #
    if nproc == 1:
        for i in range(cfg['SMART_RUN']['NUMEN']):
            print('\tEnsemble {}'.format(i+1))
            da_remapped = remap_and_save_smart_prec(
                os.path.join(prec_input_dir, 'prec_corrected.ens{}.'.format(i+1)),
                start_year, end_year,
                da_domain_target, out_remapped_dir, 'prec_corrected.ens{}.'.format(i+1),
                reuse_weight=True)
    # --- If nproc > 1, use multiprocessing --- #
    elif nproc > 1:
        # --- Set up multiprocessing --- #
        pool = mp.Pool(processes=nproc)
        # --- Loop over each ensemble member --- #
        for i in range(cfg['SMART_RUN']['NUMEN']):
            print('\tEnsemble {}'.format(i+1))
            pool.apply_async(remap_and_save_smart_prec,
                             (os.path.join(prec_input_dir, 'prec_corrected.ens{}.'.format(i+1)),
                              start_year, end_year,
                              da_domain_target, out_remapped_dir, 'prec_corrected.ens{}.'.format(i+1),
                              True))
        # --- Finish multiprocessing --- #
        pool.close()
        pool.join()



