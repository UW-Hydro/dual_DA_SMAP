
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

from da_utils import remap_con, setup_output_dirs, remap_and_save_prec


# ======================================================= #
# Parameters
# ======================================================= #
start_year = 2015
end_year = 2017
domain_target_nc = '/civil/hydro/ymao/data_assim/param/vic/ArkRed/ArkRed.domain.gpm.nc'
domain_source_nc = '/civil/hydro/ymao/data_assim/param/vic/ArkRed/ArkRed.domain.nc'
prec_input_basepath = '/civil/hydro/ymao/data_assim/forcing/vic/NLDAS-2/ArkRed/force.'  # 'YYYY.nc' will be appended

out_remapped_dir = '/civil/hydro/ymao/data_assim/forcing/vic/NLDAS-2/ArkRed.gpm_grid/'
out_file_prefix = 'force.'


# ============================================================ #
# Load input precipitation fields, remap, and save
# ============================================================ #
print('Remapping...')
# --- Load domain files --- #
da_domain_target = xr.open_dataset(domain_target_nc)['mask']
da_domain_source = xr.open_dataset(domain_source_nc)['mask']

# --- Remap and save --- #
da_remapped = remap_and_save_prec(
    prec_input_basepath,
    start_year, end_year,
    da_domain_target, out_remapped_dir, out_file_prefix,
    reuse_weight=False, da_domain_source=da_domain_source)




