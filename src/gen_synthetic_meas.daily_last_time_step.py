# =========================================================== #
# This script produces true and synthetic surface soil moisture measurements
#    - Run VIC with perturbed forcings and states --> "truth"
#    - Add random noise to "truth" top-layer soil moisture --> synthetic measurements
# =========================================================== #

import sys
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from tonic.models.vic.vic import VIC
from tonic.io import read_configobj

from da_utils import (Forcings, setup_output_dirs, propagate,
                      perturb_soil_moisture_states, concat_vic_history_files)

# =========================================================== #
# Load config file
# =========================================================== #
cfg = read_configobj(sys.argv[1])

# =========================================================== #
# Set random generation seed
# =========================================================== #
np.random.seed(cfg['CONTROL']['seed'])

# =========================================================== #
# Process some config parameters
# =========================================================== #
print('Processing config parameters...')
# Simulation time
start_time = pd.to_datetime(cfg['TIME_INDEX']['start_time'])
end_time = pd.to_datetime(cfg['TIME_INDEX']['end_time'])

# Construct forcing variable name dictionary
dict_varnames = {}
dict_varnames['PREC'] = cfg['VIC']['PREC']

# Set up output sub-directories
dirs = setup_output_dirs(os.path.join(cfg['CONTROL']['root_dir'],
                                      cfg['OUTPUT']['output_basedir']),
                         mkdirs=['truth', 'synthetic_meas', 'plots'])
truth_subdirs = setup_output_dirs(dirs['truth'],
                                  mkdirs=['global', 'history', 'states',
                                          'forcings', 'logs'])

# Construct time points for synthetic measurement (daily, last hour)
meas_times = pd.date_range(
        '{}{:02d}{:02d}-{:02d}'.format(start_time.year,
                                       start_time.month,
                                       start_time.day,
                                       cfg['TIME_INDEX']['last_hour']),
        '{}{:02d}{:02d}-{:02d}'.format(end_time.year,
                                       end_time.month,
                                       end_time.day,
                                       cfg['TIME_INDEX']['last_hour']),
        freq='D')

# VIC global template file
global_template = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['VIC']['vic_global_template'])

# =========================================================== #
# Simulate "truth" - run VIC with perturbed forcings and
# states
# =========================================================== #
print('Simulating \"truth\" - run VIC with perturbed forcings and states...')
# --- Create class VIC --- #
vic_exe = VIC(os.path.join(cfg['CONTROL']['root_dir'], cfg['VIC']['exe']))

# --- Prepare perturbed forcing data --- #
print('\tPreparing perturbed forcing data...')
start_year = start_time.year
end_year = end_time.year
for year in range(start_year, end_year+1):
    # Construct Forcings class
    class_forcings_orig = Forcings(xr.open_dataset(
                os.path.join(cfg['CONTROL']['root_dir'],
                             '{}{}.nc'.format(cfg['VIC']['orig_forcing_nc_basepath'],
                                              year))))
    # Perturb precipitaiton
    ds_perturbed = class_forcings_orig.perturb_prec_lognormal(
                                varname=dict_varnames['PREC'],
                                std=cfg['FORCINGS_STATES_PERTURB']['prec_std'])
    # Save to nc file
    ds_perturbed.to_netcdf(os.path.join(truth_subdirs['forcings'],
                                        'forc_perturbed.{}.nc'.format(year)),
                           format='NETCDF4_CLASSIC')

# --- Run VIC with perturbed forcings and soil moisture states --- #
# Initialize a list of file paths to be concatenated
list_history_paths = []

# (1) Run VIC until the first measurement time point (no initial state)
prop_period_stamp = '{}-{}'.format(start_time.strftime('%Y%m%d_%H%S'),
                                   meas_times[0].strftime('%Y%m%d_%H%S'))
print('\tRun VIC until the first measurement time {}...'.format(prop_period_stamp))
# Prepare log directories
log_dir = setup_output_dirs(
                    truth_subdirs['logs'],
                    mkdirs=['propagate.{}'.format(prop_period_stamp)])\
          ['propagate.{}'.format(prop_period_stamp)]
# Propagate until the first measurement point
propagate(start_time=start_time, end_time=meas_times[0],
          vic_exe=vic_exe, vic_global_template_file=global_template,
          vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
          init_state_nc=None,
          out_state_basepath=os.path.join(truth_subdirs['states'],
                                          'propagated.state'),
          out_history_dir=truth_subdirs['history'],
          out_history_fileprefix='history',
          out_global_basepath=os.path.join(truth_subdirs['global'], 'global'),
          out_log_dir=log_dir,
          forcing_perturbed_basepath=os.path.join(truth_subdirs['forcings'],
                                                  'forc_perturbed.'))
# Concat output history file to the list to be concatenated
list_history_paths.append(os.path.join(truth_subdirs['history'],
                                       'history.{}-{:05d}.nc'.format(
                                            start_time.strftime('%Y-%m-%d'),
                                            start_time.hour*3600+start_time.second)))

# (2) Loop over until each measurement point and run VIC
for t in range(len(meas_times)):
    # --- Determine last, current and next time point --- #
    last_time = meas_times[t]
    current_time = last_time +\
                   pd.DateOffset(hours=24/cfg['VIC']['model_steps_per_day'])
    if t == len(meas_times)-1:  # if the current time is the last measurement time
        next_time = end_time
    else:  # if not the last measurement time
        next_time = meas_times[t+1]
    # If current_time > next_time, do nothing (we already reach the end of the simulation)
    if current_time > next_time:
        break
    print('\tRun VIC ', current_time, 'to', next_time, '(perturbed forcings and states)')

    # --- Perturb states --- #
    orig_state_nc = os.path.join(
                            truth_subdirs['states'],
                            'propagated.state.{}_{:05d}.nc'.format(
                                    last_time.strftime('%Y%m%d'),
                                    last_time.hour*3600+last_time.second))
    perturbed_state_nc = os.path.join(
                            truth_subdirs['states'],
                            'perturbed.state.{}_{:05d}.nc'.format(
                                    last_time.strftime('%Y%m%d'),
                                    last_time.hour*3600+last_time.second))
    perturb_soil_moisture_states(
            states_to_perturb_nc=orig_state_nc,
            global_path=global_template,
            sigma_percent=cfg['FORCINGS_STATES_PERTURB']['state_perturb_sigma_percent'],
            out_states_nc=perturbed_state_nc)

    # --- Propagate to the next time point --- #
    propagate(start_time=current_time, end_time=next_time,
              vic_exe=vic_exe, vic_global_template_file=global_template,
              vic_model_steps_per_day=cfg['VIC']['model_steps_per_day'],
              init_state_nc=perturbed_state_nc,
              out_state_basepath=os.path.join(truth_subdirs['states'],
                                              'propagated.state'),
              out_history_dir=truth_subdirs['history'],
              out_history_fileprefix='history',
              out_global_basepath=os.path.join(truth_subdirs['global'], 'global'),
              out_log_dir=log_dir,
              forcing_perturbed_basepath=os.path.join(truth_subdirs['forcings'],
                                                      'forc_perturbed.'))
    # Concat output history file to the list to be concatenated
    list_history_paths.append(os.path.join(truth_subdirs['history'],
                                           'history.{}-{:05d}.nc'.format(
                                                current_time.strftime('%Y-%m-%d'),
                                                current_time.hour*3600+current_time.second)))

# (3) Concatenate all history files
ds_concat = concat_vic_history_files(list_history_paths)
# Save to history output directory
first_time = pd.to_datetime(ds_concat['time'][0].values)
last_time = pd.to_datetime(ds_concat['time'][-1].values)
hist_concat_nc = os.path.join(truth_subdirs['history'],
                                 'history.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                        first_time.strftime('%Y%m%d'),
                                        first_time.hour*3600+first_time.second,
                                        last_time.strftime('%Y%m%d'),
                                        last_time.hour*3600+last_time.second))
ds_concat.to_netcdf(hist_concat_nc)


## =========================================================== #
## Classes and functions
## =========================================================== #
#
#class VarToPerturb(object):
#    ''' This class is a variable to be perturbed
#
#    Atributes
#    ---------
#    da: <xarray.DataArray>
#        A dataarray of the variable to be perturbed
#
#    Require
#    ---------
#    numpy
#    '''
#
#    def __init__(self, da):
#        self.da = da
#
#    def add_gaussian_white_noise(self, sigma):
#        ''' Add Gaussian noise for all active grid cells
#
#        Parameters
#        ----------
#        '''
#        
#        # Generate random noise for the whole field
#        da_noise = self.da.copy()
#        da_noise[:] = np.random.normal(loc=0, scale=sigma, size=self.da.shape)
#        # Mask out inactive cells
#        da_noise = da_noise.where(np.isnan(self.da)==False)
#        # Add noise to the original da and return
#        return self.da + da_noise
#
## =========================================================== #
## Extract VIC output soil moisture (layer 1) at the end of
## each day, and perturb
## =========================================================== #
## Load VIC output
#ds = xr.open_dataset(cfg['OUTPUT']['vic_output_hist_path'])
#
## Resample surface sm to daily time step (use the value of the last time step in each day)
#da_sm1_true = ds['OUT_SOIL_MOIST'].sel(nlayer=0)
#da_sm1_true_daily = da_sm1_true.resample(dim='time', freq='D', how='last')
#
## Reformat time index
#da_sm1_true_daily['time'] = pd.date_range(
#        '{}-{:2d}'.format(cfg['TIME_INDEX']['start_date'],
#                          cfg['TIME_INDEX']['last_hour']),
#        '{}-{:2d}'.format(cfg['TIME_INDEX']['end_date'],
#                          cfg['TIME_INDEX']['last_hour']),
#        freq='D')
#
## Add noise
#VarToPerturb_sm1 = VarToPerturb(da_sm1_true_daily) # create class
#da_sm1_perturbed = VarToPerturb_sm1.add_gaussian_white_noise(cfg['NOISE_SIM']['sigma']) # add noise
#
## Add attributes to the simulated measurements
#da_sm1_perturbed.attrs['units'] = 'mm'
#da_sm1_perturbed.attrs['long_name'] = 'Simulated surface soil moisture measurement'
#
## =========================================================== #
## Write the simulated measurement to netCDF file
## =========================================================== #
#ds_simulated = xr.Dataset({'simulated_surface_sm': da_sm1_perturbed})
#ds_simulated.to_netcdf(cfg['OUTPUT']['output_sim_meas'], format='NETCDF4_CLASSIC')
#
## =========================================================== #
## Plot - compare orig. and simulated sm1, daily
## =========================================================== #
#fig = plt.figure()
#plt.plot(da_sm1_true_daily.squeeze(), label='Orig. VIC output')
#plt.plot(da_sm1_perturbed.squeeze(), label='Simulated meas. (perturbed)')
#plt.legend()
#plt.xlabel('Day')
#plt.ylabel('Soil moisture (mm)')
#plt.title('Surface soil moisture')
#fig.savefig(cfg['OUTPUT']['output_plot_path'], format='png')
#
#
