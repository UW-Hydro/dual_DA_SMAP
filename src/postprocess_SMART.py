
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

from tonic.io import read_configobj

from da_utils import (load_nc_and_concat_var_years, setup_output_dirs,
                      da_2D_to_3D_from_SMART, da_3D_to_2D_for_SMART,
                      correct_prec_from_SMART, rmse)


# ============================================================ #
# Process command line arguments
# Read config file
# ============================================================ #
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Process some input variables
# ============================================================ #
start_date = pd.datetime.strptime(cfg['SMART_RUN']['start_date'], "%Y-%m-%d")
end_date = pd.datetime.strptime(cfg['SMART_RUN']['end_date'], "%Y-%m-%d")
start_year = start_date.year
end_year = end_date.year


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
                  ['prec_orig']
    
# --- Load corrected window-averaged prec --- #
run_SMART_outfile = os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['OUTPUT']['output_basedir'],
                                 'run_SMART',
                                 'SMART_corrected_rainfall.mat')
run_SMART_prec_corr = loadmat(run_SMART_outfile)['RAIN_SMART_SMOS']  # [nwindow, npixel]

# Load in domain file
ds_domain = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['DOMAIN']['domain_file']))
da_mask = ds_domain['mask']

# Preprocess SMART output prec
dict_prec_SMART_window = {'prec_corr_window': run_SMART_prec_corr}
nwindow = run_SMART_prec_corr.shape[0]
da_prec_corr_window = da_2D_to_3D_from_SMART(
                            dict_array_2D=dict_prec_SMART_window,
                            da_mask=da_mask,
                            out_time_varname='window',
                            out_time_coord=range(nwindow))['prec_corr_window']


# ============================================================ #
# Rescale orig. prec at orig. timestep based on SMART outputs
# ============================================================ #
print('Rescaling original prec...')

da_prec_corrected = correct_prec_from_SMART(
                            da_prec_orig,
                            cfg['SMART_RUN']['window_size'],
                            da_prec_corr_window,
                            start_date)


# ============================================================ #
# Save final corrected prec data to netCDF file
# ============================================================ #
print('Saving corrected prec data to netCDF files...')

# Put da to ds
ds_prec_corrected = xr.Dataset({'prec_corrected': da_prec_corrected})

# Set up output subdir
out_dir = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['OUTPUT']['output_basedir']),
                            mkdirs=['post_SMART'])['post_SMART']

# Save to netCDF file; separate files for each year ending in 'YYYY.nc'
for year, ds in ds_prec_corrected.groupby('time.year'):
    ds.to_netcdf(os.path.join(out_dir, 'prec_corrected.{}.nc'.format(year)))

