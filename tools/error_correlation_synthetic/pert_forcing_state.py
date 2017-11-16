import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

from tonic.io import read_config, read_configobj

from da_utils import (Forcings, perturb_forcings_ensemble, setup_output_dirs,
                      to_netcdf_forcing_file_compress, calculate_sm_noise_to_add_magnitude,
                      calculate_scale_n_whole_field, to_netcdf_state_file_compress,
                      calculate_max_soil_moist_domain, convert_max_moist_n_state)
from error_utils import (load_nc_file, pert_prec_state_cell_ensemble)


# ============================================================ #
# Process command line arguments
# ============================================================ #
parser = argparse.ArgumentParser()
parser.add_argument('--cfg',
                    help='Config file')
parser.add_argument('--corrcoef', type=float,
                    help='Correlation coefficient of perturbation')
parser.add_argument('--phi', type=float,
                    help='Autocorrelation parameter of AR(1) for both state and forcing perturbation')
parser.add_argument('--N', type=int,
                    help='Ensemble size for perturbation')
parser.add_argument('--nproc', type=int, default=1,
                    help='Number of processors to use')
args = parser.parse_args()

# Read config file
cfg = read_configobj(args.cfg)
# Correlation coefficient of perturbation
corrcoef = args.corrcoef
# Autocorrelation parameter of AR(1) for both state and forcing perturbation
phi = args.phi
# Perturbation ensemble size
N = args.N
# Number of processors to use
nproc = args.nproc


# ===================================================== #
# Parameter setting
# ===================================================== #
# Root directory - all other paths will be under root_dir
root_dir = cfg['CONTROL']['root_dir']

# --- Time --- #
start_time = pd.to_datetime(cfg['TIME']['start_time'])
end_time = pd.to_datetime(cfg['TIME']['end_time'])
start_year = start_time.year
end_year = end_time.year
state_times = pd.date_range(start_time, end_time, freq='D')

# --- Input forcings and states --- #
# Orig. forcing netcdf basepath ('YYYY.nc' will be appended)
force_orig_nc = os.path.join(root_dir, cfg['INPUTS']['force_orig_basepath'])
# Orig. history netcdf file
hist_orig_nc = os.path.join(root_dir, cfg['INPUTS']['hist_orig_path'])
# Orig state basepath ('YYYYMMDD_SSSSS.nc' will be appended); initial state not included
state_orig_basepath = os.path.join(root_dir, cfg['INPUTS']['state_orig_basepath'])
# Initial state file
init_state_nc = os.path.join(root_dir, cfg['INPUTS']['init_state_path'])
# VIC global template file
vic_global_template_path = os.path.join(root_dir, cfg['INPUTS']['vic_global_template_path'])

# --- Perturbation --- #
### prec. perturbation ###
# Standard deviation of prec. perturbation multiplier (fixed)
prec_std = cfg['PERTURB']['prec_std']
# Parameter in AR(1) process for prec. perturbation (fixed)
# prec_phi = 0
### State perturbation ###
# Percentage of max value of each state to perturb (e.g., if
# state_perturb_sigma_percent = 5, then Gaussian noise with standard deviation
# = 5% of max soil moisture will be added as perturbation)
# state_perturb_sigma_percent is a list of percentage for each soil layer
# (e.g., if there are three soil layers, then an example of state_perturb_sigma_percent is: 20,1,0.5)
# NOTE: keep this consistent as in the dual correction system
state_perturb_sigma_percent = cfg['PERTURB']['state_perturb_sigma_percent']

# --- Outputs --- #
output_dir = cfg['OUTPUT']['output_dir']


# ===================================================== #
# Load forcings and prepare 
# ===================================================== #
print('Loading orig. forcings...')
# --- Load orig. forcing --- #
ds_force_orig = load_nc_file(force_orig_nc + '{}.nc', start_year, end_year)
# Select indicated time range
ds_force_orig = ds_force_orig.sel(time=slice(start_time, end_time))
# --- Set up output directory --- #
force_pert_basedir = setup_output_dirs(
    os.path.join(root_dir, output_dir),
    mkdirs=['perturbed_forcings'])['perturbed_forcings']
force_pert_noise_basedir = setup_output_dirs(
    force_pert_basedir,
    mkdirs=['corrcoef_{}_phi_{}'.format(corrcoef, phi)])\
    ['corrcoef_{}_phi_{}'.format(corrcoef, phi)]


# ===================================================== #
# Load state files and prepare
# ===================================================== #
print('Loading orig. states...')
# Load states
states_orig = []
# Load initial state
states_orig.append(xr.open_dataset(init_state_nc))
# Load all times of states
for time in state_times[1:]:
    state_nc = state_orig_basepath + time.strftime('%Y%m%d') + '_' + '{:05d}'.format(time.second) + '.nc'
    states_orig.append(xr.open_dataset(state_nc))

# --- Set up output directory --- #
state_pert_basedir = setup_output_dirs(
    os.path.join(root_dir, output_dir),
    mkdirs=['perturbed_states'])['perturbed_states']
state_pert_noise_basedir = setup_output_dirs(
    state_pert_basedir,
    mkdirs=['corrcoef_{}_phi_{}'.format(corrcoef, phi)])['corrcoef_{}_phi_{}'.format(corrcoef, phi)]

# --- Prepare for state perturbation --- #
# --- (Same perturbation for all tiles/layers, with different magtinude for each layer) --- #
# Calculate perturbation scale for each layer
da_scale = calculate_sm_noise_to_add_magnitude(
    vic_history_path=hist_orig_nc,
    sigma_percent=state_perturb_sigma_percent)
# Extract maximum moisture for each layer
nveg = len(states_orig[0]['veg_class'])
nsnow = len(states_orig[0]['snow_band'])
da_max_moist = calculate_max_soil_moist_domain(vic_global_template_path)
da_max_moist_n = convert_max_moist_n_state(da_max_moist, nveg, nsnow)
scale_n_nloop = calculate_scale_n_whole_field(
    da_scale, nveg, nsnow)  # [lat*lon=1, n=nlayer*nveg*nsnow]


# ===================================================== #
# Add perturbation to states & forcings
# ===================================================== #
np.random.seed(1111)
# Add perturbation to generate an ensemble of perturbed states & forcings
# (same multiplier for prec every day)
pert_prec_state_cell_ensemble(
    N=N, state_times=state_times, corrcoef=corrcoef, phi=phi,
    ds_force_orig=ds_force_orig, prec_std=prec_std,
    out_forcing_basedir=force_pert_noise_basedir,
    states_orig=states_orig, scale_n_nloop=scale_n_nloop,
    out_state_basedir=state_pert_noise_basedir, da_max_moist_n=da_max_moist_n,
    nproc=nproc)


