
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import sys

from tonic.io import read_config, read_configobj


# ============================================================ #
# Process command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])


# ============================================================ #
# Load data
# ============================================================ #
print('Loading data...')

# --- Determine the output directories --- #
dirs = OrderedDict()
out_basedir = os.path.join(cfg['CONTROL']['root_dir'],
                           cfg['OUTPUT']['output_basdir'])
dirs['history'] = os.path.join(out_basedir, 'history')
dirs['plots'] = os.path.join(out_basedir, 'plots')

# --- Load truth history file --- #
print('\tLoading truth')
ds_truth = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                        cfg['EnKF']['truth_hist_nc']))

# --- Load measurement data --- #
print('\tLoading measurement')
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

# --- Load open-loop history file --- #
print('\tLoading open-loop')
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])
hist_openloop_nc = os.path.join(
                        dirs['history'],
                        'history.openloop.{}-{:05d}.nc'.format(
                                start_time.strftime('%Y-%m-%d'),
                                start_time.hour*3600+start_time.second))
ds_openloop = xr.open_dataset(hist_openloop_nc)

# --- Load EnKF post-processed ensemble mean results --- #
print('\tLoading ensemble mean')
hist_ens_mean_post = os.path.join(
                        dirs['history'],
                        'postprocess_ens_mean_updated_states',
                        'history.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                start_time.strftime('%Y%m%d'),
                                start_time.hour*3600+start_time.second,
                                end_time.strftime('%Y%m%d'),
                                end_time.hour*3600+end_time.second))
ds_EnKF_ens_mean = xr.open_dataset(hist_ens_mean_post)

# --- Load all ensemble member results --- #
print('\tLoading each ensemble member')
dict_ens_ds = {}
for i in range(cfg['EnKF']['N']):
    fname = os.path.join(
                    dirs['history'],
                    'EnKF_ensemble_concat',
                    'history.ens{}.concat.{}_{:05d}-{}_{:05d}.nc'.format(
                                i+1,
                                start_time.strftime('%Y%m%d'),
                                start_time.hour*3600+start_time.second,
                                end_time.strftime('%Y%m%d'),
                                end_time.hour*3600+end_time.second))
    dict_ens_ds['ens{}'.format(i+1)] =  xr.open_dataset(fname)


# ============================================================ #
# Plot EnKF results
# ============================================================ #
# Extract top layer soil moisture from all datasets
da_sm1_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0)
da_sm1_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_SOIL_MOIST'].sel(nlayer=0)
da_sm1_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0)
dict_ens_da_sm1 = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
    dict_ens_da_sm1[ens_name] = dict_ens_ds[ens_name]['OUT_SOIL_MOIST'].sel(nlayer=0)

# Extract lat's and lon's
lat = da_sm1_openloop['lat'].values
lon = da_sm1_openloop['lon'].values

# ------------------------------------------------------------ #
# Plot results - soil moisture time series
# ------------------------------------------------------------ #
print('Plotting top-layer soil moisture...')
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm1_openloop.loc[da_sm1_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue

        print('\t lat {}, lon {}'.format(lt, lg))
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_da_sm1[ens_name].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        da_sm1_EnKF_ens_mean.loc[:, lt, lg].to_series().plot(
                                                color='b', style='-',
                                                label='EnKF ens. mean', legend=True)
        # plot measurement
        da_meas.loc[:, lt, lg, 0].to_series().plot(
                        style='ro', label='Measurement',
                        legend=True)
        # plot truth
        da_sm1_truth.loc[:, lt, lg].to_series().plot(color='k', style='-',
                                                     label='Truth', legend=True)
        # plot open-loop
        da_sm1_openloop.loc[:, lt, lg].to_series().plot(color='m', style='--',
                                                        label='Open-loop', legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Soil moiture (mm)')
        plt.title('Top-layer soil moisture, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'sm1_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # Save figure for a shorter period
        plt.xlim([pd.datetime(1949, 5, 10), pd.datetime(1949, 5, 30)])
        fig.savefig(os.path.join(dirs['plots'], 'sm1_{}_{}.shorter.png'.format(lt, lg)),
                    format='png')
        
# ------------------------------------------------------------ #
# Plot results - surface runoff
# ------------------------------------------------------------ #
print('Plotting surface runoff...')
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm1_openloop.loc[da_sm1_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue

        print('\t lat {}, lon {}'.format(lt, lg))
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_ds[ens_name]['OUT_RUNOFF'].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ds_EnKF_ens_mean['OUT_RUNOFF'].loc[:, lt, lg].to_series().plot(
                                                color='b', style='-',
                                                label='EnKF ens. mean', legend=True)
        # plot measurement
        # da_meas.loc[:, lt, lg, 0].to_series().plot(
        #                style='ro', label='Measurement',
        #                legend=True)
        # plot truth
        ds_truth['OUT_RUNOFF'].loc[:, lt, lg].to_series().plot(color='k', style='-',
                                                     label='Truth', legend=True)
        # plot open-loop
        ds_openloop['OUT_RUNOFF'].loc[:, lt, lg].to_series().plot(color='m', style='--',
                                                        label='Open-loop', legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Runoff (mm)')
        plt.title('Surface runoff, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'runoff_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # Save figure for a shorter period
        plt.xlim([pd.datetime(1949, 1, 1), pd.datetime(1949, 1, 10)])
        ylim = ds_truth['OUT_RUNOFF'].loc[:, lt, lg].to_series().truncate(before=pd.datetime(1949, 1, 1),
                                                                          after=pd.datetime(1949, 1, 10)).max() * 2
        plt.ylim([0, ylim])
        fig.savefig(os.path.join(dirs['plots'], 'runoff_{}_{}.shorter.png'.format(lt, lg)),
                    format='png')
        
# ------------------------------------------------------------ #
# Plot results - baseflow
# ------------------------------------------------------------ #
print('Plotting baseflow...')
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm1_openloop.loc[da_sm1_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue

        print('\t lat {}, lon {}'.format(lt, lg))
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_ds[ens_name]['OUT_BASEFLOW'].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ds_EnKF_ens_mean['OUT_BASEFLOW'].loc[:, lt, lg].to_series().plot(
                                                color='b', style='-',
                                                label='EnKF ens. mean', legend=True)
        # plot measurement
        # da_meas.loc[:, lt, lg, 0].to_series().plot(
        #                style='ro', label='Measurement',
        #                legend=True)
        # plot truth
        ds_truth['OUT_BASEFLOW'].loc[:, lt, lg].to_series().plot(color='k', style='-',
                                                     label='Truth', legend=True)
        # plot open-loop
        ds_openloop['OUT_BASEFLOW'].loc[:, lt, lg].to_series().plot(color='m', style='--',
                                                        label='Open-loop', legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Baseflow (mm)')
        plt.title('Baseflow, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'baseflow_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # Save figure for a shorter period
        plt.xlim([pd.datetime(1949, 5, 10), pd.datetime(1949, 5, 30)])
        ylim = ds_truth['OUT_BASEFLOW'].loc[:, lt, lg].to_series().truncate(before=pd.datetime(1949, 5, 10),
                                                                          after=pd.datetime(1949, 5, 30)).max() * 2
        plt.ylim([0, ylim])
        fig.savefig(os.path.join(dirs['plots'], 'baseflow_{}_{}.shorter.png'.format(lt, lg)),
                    format='png')
        
