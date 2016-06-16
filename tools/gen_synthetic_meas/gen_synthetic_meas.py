'''
This script simulates surface soil moisture measurements
    - by perturbing VIC-simulated top-layer soil moisture by Gaussian white
      noise
'''

import sys
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from tonic.models.vic.vic import VIC
from tonic.io import read_configobj

# =========================================================== #
# Classes and functions
# =========================================================== #

class VarToPerturb(object):
    ''' This class is a variable to be perturbed

    Atributes
    ---------
    da: <xarray.DataArray>
        A dataarray of the variable to be perturbed

    Require
    ---------
    numpy
    '''

    def __init__(self, da):
        self.da = da

    def add_gaussian_white_noise(self, sigma):
        ''' Add Gaussian noise for all active grid cells

        Parameters
        ----------
        '''
        
        # Generate random noise for the whole field
        da_noise = self.da.copy()
        da_noise[:] = np.random.normal(loc=0, scale=sigma, size=self.da.shape)
        # Mask out inactive cells
        da_noise = da_noise.where(np.isnan(self.da)==False)
        # Add noise to the original da and return
        return self.da + da_noise


# =========================================================== #
# Load config file
# =========================================================== #
cfg = read_configobj(sys.argv[1])

# =========================================================== #
# Set random generation seed
# =========================================================== #
np.random.seed(cfg['CONTROL']['seed'])

# =========================================================== #
# Run VIC
# =========================================================== #
# Create class VIC
vic_exe = VIC(cfg['VIC']['exe'])
# Run VIC
vic_exe.run(cfg['VIC']['global'],
            logdir=cfg['OUTPUT']['vic_log_dir'])

# =========================================================== #
# Extract VIC output soil moisture (layer 1) at the end of
# each day, and perturb
# =========================================================== #
# Load VIC output
ds = xr.open_dataset(cfg['OUTPUT']['vic_output_hist_path'])

# Resample surface sm to daily mean
da_sm1_true = ds['OUT_SOIL_MOIST'].sel(nlayer=0)
da_sm1_true_daily = da_sm1_true.resample(dim='time', freq='D', how='mean')

# Reset time index to noon on each day
da_sm1_true_daily['time'] = pd.date_range(
        '{}-12'.format(cfg['TIME_INDEX']['start_date']),
        '{}-12'.format(cfg['TIME_INDEX']['end_date']),
        freq='D')

# Add noise
VarToPerturb_sm1 = VarToPerturb(da_sm1_true_daily) # create class
da_sm1_perturbed = VarToPerturb_sm1.add_gaussian_white_noise(cfg['NOISE_SIM']['sigma']) # add noise

# Add attributes to the simulated measurements
da_sm1_perturbed.attrs['units'] = 'mm'
da_sm1_perturbed.attrs['long_name'] = 'Simulated surface soil moisture measurement'

# =========================================================== #
# Write the simulated measurement to netCDF file
# =========================================================== #
ds_simulated = xr.Dataset({'simulated_surface_sm': da_sm1_perturbed})
ds_simulated.to_netcdf(cfg['OUTPUT']['output_sim_meas'], format='NETCDF4_CLASSIC')

# =========================================================== #
# Plot - compare orig. and simulated sm1, daily
# =========================================================== #
fig = plt.figure()
plt.plot(da_sm1_true_daily.squeeze(), label='Orig. VIC output')
plt.plot(da_sm1_perturbed.squeeze(), label='Simulated meas. (perturbed)')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Soil moisture (mm)')
plt.title('Surface soil moisture')
fig.savefig(cfg['OUTPUT']['output_plot_path'], format='png')


