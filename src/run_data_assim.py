
import sys
import numpy as np
import xarray as xr
import os
import pandas as pd

from tonic.models.vic.vic import VIC
from tonic.io import read_config, read_configobj
from da_utils import EnKF_VIC, generate_VIC_global_file, setup_output_dirs

# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# ============================================================ #
# Set random generation seed
# ============================================================ #
np.random.seed(cfg['CONTROL']['seed'])

# ============================================================ #
# Prepare output directories
# ============================================================ #
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_basdir']),
                         mkdirs=['global', 'history', 'states', 'logs', 'plots'])

# ============================================================ #
# Pre-processing
# ============================================================ #
P0 = np.diag(np.asarray(cfg['EnKF']['P0']))
x0 = np.asarray(cfg['EnKF']['x0'])

# ============================================================ #
# Prepare and run EnKF
# ============================================================ #
# --- Prepare initial states and error matrix --- #
P0 = np.diag(np.asarray(cfg['EnKF']['P0']))
x0 = np.asarray(cfg['EnKF']['x0'])

# --- Load and process measurement data --- #
# Load measurement data
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
# Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(cfg['EnKF']['start_time'], cfg['EnKF']['end_time']))


EnKF_VIC(N=cfg['EnKF']['N'],
         start_time=pd.to_datetime(cfg['EnKF']['start_time']),
         end_time=pd.to_datetime(cfg['EnKF']['end_time']),
         P0=P0,
         x0=x0,
         da_meas=da_meas,
         da_meas_time_var='time',
         vic_global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['VIC']['vic_global_template']),
         vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
         output_vic_global_dir=dirs['global'],
         output_vic_state_dir=dirs['states'],
         output_vic_result_dir=dirs['history'])


