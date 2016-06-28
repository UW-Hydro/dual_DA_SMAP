
import sys
import numpy as np
import xarray as xr
import os
import pandas as pd
from collections import OrderedDict

from tonic.models.vic.vic import VIC
from tonic.io import read_config, read_configobj
from da_utils import EnKF_VIC, setup_output_dirs, generate_VIC_global_file

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
                         mkdirs=['global', 'history', 'states', 'forcings',
                                 'logs', 'plots'])

# ============================================================ #
# Prepare VIC exe
# ============================================================ #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['vic_exe']))

# ============================================================ #
# VIC model spinup
# VIC will be run for a spinup period to get x0
# ============================================================ #
# Determine VIC spinup run period
vic_run_start_time = pd.to_datetime(cfg['VIC']['spinup_start_time'])
vic_run_end_time = pd.to_datetime(cfg['EnKF']['start_time'])

# Generate VIC global param file (no initial state)
replace = OrderedDict([('FORCING1',
                        os.path.join(cfg['CONTROL']['root_dir'],
                                     cfg['FORCINGS']['orig_forcing_nc_basepath']))])
global_file = generate_VIC_global_file(
                        global_template_path=os.path.join(cfg['CONTROL']['root_dir'],
                                                          cfg['VIC']['vic_global_template']),
                        model_steps_per_day=cfg['VIC']['model_steps_per_day'],
                        start_time=vic_run_start_time,
                        end_time=vic_run_end_time,
                        init_state="# INIT_STATE",
                        vic_state_basepath=os.path.join(dirs['states'], 'state.spinup'),
                        vic_history_file_basepath=os.path.join(dirs['history'],
                                                               'history.spinup'),
                        replace=replace,
                        output_global_basepath=os.path.join(dirs['global'],
                                                            'global.spinup'))

# Run VIC
#returncode = vic_exe.run(global_file, logdir=output_vic_log_dir)
#check_returncode(returncode, expected=0)

# ============================================================ #
# Prepare and run EnKF
# ============================================================ #
# --- Load and process measurement data --- #
# Load measurement data
ds_meas_orig = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                            cfg['EnKF']['meas_nc']))
da_meas_orig = ds_meas_orig[cfg['EnKF']['meas_var_name']]
# Only select out the period within the EnKF run period
da_meas = da_meas_orig.sel(time=slice(cfg['EnKF']['start_time'], cfg['EnKF']['end_time']))
# Convert da_meas dimension to [time, lat, lon, m] (currently m = 1)
time = da_meas['time']
lat = da_meas['lat']
lon = da_meas['lon']
data = da_meas.values.reshape((len(time), len(lat), len(lon), 1))
da_meas = xr.DataArray(data, coords=[time, lat, lon, [0]],
                       dims=['time', 'lat', 'lon', 'm'])

# --- Process VIC forcing names and perturbation parameters --- #
# Construct forcing variable name dictionary
dict_varnames = {}
dict_varnames['PREC'] = cfg['FORCINGS']['PREC']

# --- Prepare measurement error covariance matrix R [m*m] --- #
R = np.array([[cfg['EnKF']['R']]])


EnKF_VIC(N=cfg['EnKF']['N'],
         start_time=pd.to_datetime(cfg['EnKF']['start_time']),
         end_time=pd.to_datetime(cfg['EnKF']['end_time']),
         init_state_basepath=os.path.join(dirs['states'], 'state.spinup'),
         P0=cfg['EnKF']['P0'],
         R=R,
         da_meas=da_meas,
         da_meas_time_var='time',
         vic_exe=vic_exe,
         vic_global_template=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['VIC']['vic_global_template']),
         vic_forcing_orig_basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['FORCINGS']['orig_forcing_nc_basepath']),
         vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
         output_vic_global_root_dir=dirs['global'],
         output_vic_state_root_dir=dirs['states'],
         output_vic_history_root_dir=dirs['history'],
         output_vic_forcing_root_dir=dirs['forcings'],
         output_vic_log_root_dir=dirs['logs'],
         dict_varnames=dict_varnames, prec_std=cfg['FORCINGS']['prec_std'],
         state_perturb_sigma_percent=cfg['EnKF']['state_perturb_sigma_percent'])


