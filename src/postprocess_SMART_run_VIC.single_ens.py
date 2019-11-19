
''' This script run VIC with SMART-corrected rainfall

    Usage:
        $ python postprocess_EnKF.py <config_file> <mpi_proc> <ens>
'''

from collections import OrderedDict
import numpy as np
import pandas as pd
import os
import xarray as xr
import sys
import multiprocessing as mp
import shutil

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic import VIC
from da_utils import (setup_output_dirs, run_vic_assigned_states,
                      Forcings, to_netcdf_forcing_file_compress, propagate)


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# Read number of processors for VIC MPI runs
mpi_proc = int(sys.argv[2])

# Ensemble index of prec to postprocess (index starts from 1)
# Options:
#   integer index of SMART-corrected prec (prec file name should be: "prec_corrected.ens<i>.YYYY.nc");
#   "mean" for SMART-corrected ensemble-mean (prec file name: "prec_corrected.YYYY.nc")
ens_prec = sys.argv[3]


# ============================================================ #
# Prepare VIC exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['RUN_VIC']['vic_exe']))


# ============================================================ #
# Check whether REMAP section is in the cfg file
# ============================================================ #
if 'RUN_VIC' in cfg:
    pass
else:
    raise ValueError('Must have [REMAP] section in the cfg file to post-regrid'
                     'SMART-corrected rainfall field!')


# ============================================================ #
# Process cfg data
# ============================================================ #
start_time = pd.to_datetime(cfg['SMART_RUN']['start_time'])
end_time = pd.to_datetime(cfg['SMART_RUN']['end_time'])

start_year = start_time.year
end_year = end_time.year


# ============================================================ #
# Setup postprocess output directories
# ============================================================ #
# Identify which SMART postprocessed directory to read from
if cfg['RUN_VIC']['smart_output_from'] == 'post':
    smart_outdir = os.path.join(
        cfg['CONTROL']['root_dir'], cfg['OUTPUT']['output_basedir'], 'post_SMART')
elif cfg['RUN_VIC']['smart_output_from'] == 'spatial_downscale':
    smart_outdir = os.path.join(
        cfg['CONTROL']['root_dir'], cfg['OUTPUT']['output_basedir'], 'post_spatial_downscaled')
elif cfg['RUN_VIC']['smart_output_from'] == 'remap':
    smart_outdir = os.path.join(
        cfg['CONTROL']['root_dir'], cfg['OUTPUT']['output_basedir'], 'post_final_remapped')

if 'run_vic_subdir' in cfg['RUN_VIC']:
    run_vic_subdir = cfg['RUN_VIC']['run_vic_subdir']
else:
    run_vic_subdir = 'run_vic'
output_basedir = setup_output_dirs(
                         smart_outdir,
                         mkdirs=[run_vic_subdir])[run_vic_subdir]

dirs = setup_output_dirs(output_basedir,
                         mkdirs=['global', 'history', 'forcings',
                                 'logs', 'plots'])


# ============================================================ #
# Generate forcings for vic run - combine SMART-corrected
# precip with other met variables
# ============================================================ #
# ----------------------------------------------------------------- #
print('Replacing precip data...')
# Set flag for whether to delete the forcing file after running
forcing_delete = 0
# Replace prec forcing
for year in range(start_year, end_year+1):
    # Load prec data
    if ens_prec == 'mean':
        da_prec = xr.open_dataset(
            os.path.join(smart_outdir,
                         'prec_corrected.{}.nc'.format(year)))\
            ['prec_corrected']
    else:
        da_prec = xr.open_dataset(
            os.path.join(smart_outdir,
                         'prec_corrected.ens{}.{}.nc'.format(ens_prec, year)))\
            ['prec_corrected']
    # Load in orig forcings
    class_forcings_orig = Forcings(xr.open_dataset('{}{}.nc'.format(
                os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['RUN_VIC']['orig_forcing_nc_basepath']),
                year)))
    # Select SMART time period only
    class_forcings_orig.ds = class_forcings_orig.ds.sel(time=slice(start_time, end_time))
    # Replace prec
    ds_prec_replaced = class_forcings_orig.replace_prec(
                            'PREC',
                            da_prec)
    # Save replaced forcings to netCDF file
    if ens_prec == 'mean':
        vic_forcing_basepath = os.path.join(
                dirs['forcings'],
                'forc.post_prec.ens_mean.') 
    else:
        vic_forcing_basepath = os.path.join(
                dirs['forcings'],
                'forc.post_prec.ens{}.'.format(
                    ens_prec))
    to_netcdf_forcing_file_compress(
        ds_prec_replaced,
        out_nc='{}{}.nc'.format(vic_forcing_basepath, year))
# Set flag to delete forcing after VIC run
#forcing_delete = 1

# ----------------------------------------------------------------- #
# --- Run VIC with corrected prec forcing --- #
# ----------------------------------------------------------------- #
print('Run VIC with SMART-corrected forcing...')
# --- Prepare some variables --- #
# initial state nc
init_state_nc = os.path.join(
    cfg['CONTROL']['root_dir'],
    cfg['RUN_VIC']['vic_init_state_nc'])
# make subdirs for global, history and log files for
# each ensemble member
if ens_prec == 'mean':
    subdir_name = 'force_mean'
else:
    subdir_name = 'force_ens{}'.format(ens_prec)
hist_subdir = setup_output_dirs(
                    dirs['history'],
                    mkdirs=[subdir_name])\
              [subdir_name]
global_subdir = setup_output_dirs(
                    dirs['global'],
                    mkdirs=[subdir_name])\
              [subdir_name]
log_subdir = setup_output_dirs(
                    dirs['logs'],
                    mkdirs=[subdir_name])\
              [subdir_name]
# other variables
global_template = os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['RUN_VIC']['vic_global_template'])
# --- run VIC --- #
propagate(start_time=start_time, end_time=end_time,
          vic_exe=vic_exe, vic_global_template_file=global_template,
          vic_model_steps_per_day=cfg['RUN_VIC']['model_steps_per_day'],
          init_state_nc=init_state_nc,
          out_state_basepath=None,
          out_history_dir=hist_subdir,
          out_history_fileprefix='history',
          out_global_basepath=os.path.join(global_subdir, 'global'),
          out_log_dir=log_subdir,
          forcing_basepath=vic_forcing_basepath,
          mpi_proc=mpi_proc,
          mpi_exe=cfg['RUN_VIC']['mpi_exe'],
          delete_log=False)

# --- Clean up forcing files --- #
if forcing_delete == 1:
    for year in range(start_year, end_year+1):
        f = '{}{}.nc'.format(vic_forcing_basepath, year)
        os.remove(f)


