
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xarray as xr
import os
from collections import OrderedDict
import pandas as pd
import numpy as np
import sys
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh

from tonic.io import read_config, read_configobj
from da_utils import rmse


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
# Extract lat's and lon's
lat = ds_openloop['lat'].values
lon = ds_openloop['lon'].values

# ------------------------------------------------------------ #
# Plot results - top-layer soil moisture time series
# ------------------------------------------------------------ #
print('Plotting top-layer soil moisture...')
# Extract top layer soil moisture from all datasets
da_sm1_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=0)
da_sm1_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_SOIL_MOIST'].sel(nlayer=0)
da_sm1_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=0)
dict_ens_da_sm1 = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
# Plot
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm1_openloop.loc[da_sm1_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue

        print(lt, lg)
        
        # --- RMSE --- #
        # extract time series
        ts_meas = da_meas.loc[:, lt, lg, 0].to_series()
        ts_truth = da_sm1_truth.loc[:, lt, lg].to_series()
        ts_EnKF_mean = da_sm1_EnKF_ens_mean.loc[:, lt, lg].to_series()
        ts_openloop = da_sm1_openloop.loc[:, lt, lg].to_series()
        # Calculate meas vs. truth
        df_truth_meas = pd.concat([ts_truth, ts_meas], axis=1, keys=['truth', 'meas']).dropna()
        rmse_meas = rmse(df_truth_meas, var_true='truth', var_est='meas')
        # Calculate EnKF_mean vs. truth
        df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
        rmse_EnKF_mean = rmse(df_truth_EnKF, var_true='truth', var_est='EnKF_mean')
        # Calculate open-loop vs. truth
        df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
        rmse_openloop = rmse(df_truth_openloop, var_true='truth', var_est='openloop')

        # ----- Regular plots ----- #
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
        # plot ensemble mean and standard deviation
        times = dict_ens_da_sm1['ens1']['time']
        data = np.empty([len(times), cfg['EnKF']['N']])
        columns = []
        for i in range(cfg['EnKF']['N']):
            data[:,i] = dict_ens_da_sm1['ens{}'.format(i+1)].loc[:, lt, lg].to_series().values
            columns.append('ens{}'.format(i+1))
        df = pd.DataFrame(data, index=times, columns=columns)  # put all ensemble results into a df
        # --- ensemble mean --- #
        s_mean = df.mean(axis=1)
        s_mean.plot(color='cyan', style='-',
                    label='Ensemble mean', legend=True)
        # --- ensemble standard deviation --- #
        s_std = df.std(axis=1)
        (s_mean + s_std).plot(color='cyan', style='--',
                              label='Ensemble std', legend=True)
        (s_mean - s_std).plot(color='cyan', style='--',
                              legend=False)
        # plot EnKF post-processed ens. mean
        ts_EnKF_mean.plot(color='b', style='-',
                          label='EnKF ens. mean, RMSE={:.2f}'.format(rmse_EnKF_mean),
                          legend=True)    
        # plot measurement
        ts_meas.plot(style='ro', label='Measurement, RMSE={:.2f}'.format(rmse_meas),
                     legend=True)
        # plot truth
        ts_truth.plot(color='k', style='-', label='Truth', legend=True)
        # plot open-loop
        ts_openloop.plot(color='m', style='--',
                         label='Open-loop, RMSE={:.2f}'.format(rmse_openloop),
                         legend=True)
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

        # ----- Interactive version ----- #
        # Create figure
        output_file(os.path.join(dirs['plots'], 'sm1_{}_{}.html'.format(lt, lg)))

        p = figure(title='Top-layer soil moisture, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="Soil moiture (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend="Ensemble members"
            else:
                legend=False
            ts = dict_ens_da_sm1[ens_name].loc[:, lt, lg].to_series()
            p.line(ts.index, ts.values, color="grey", line_dash="solid", alpha=0.3, legend=legend)
        # plot ensemble mean and standard deviation
        # --- ensemble mean --- #
        ts = s_mean
        p.line(ts.index, ts.values, color="cyan", line_dash="solid",
               legend="Ensemble mean", line_width=1)
        ts = s_mean + s_std
        p.line(ts.index, ts.values, color="cyan", line_dash="dashed",
               legend="Ensemble std", line_width=1)
        ts = s_mean - s_std
        p.line(ts.index, ts.values, color="cyan", line_dash="dashed",
               legend=False, line_width=1)
        # plot EnKF post-processed ens. mean
        ts = ts_EnKF_mean
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="EnKF ens. mean, RMSE={:.2f}".format(rmse_EnKF_mean), line_width=2)
        # plot measurement
        ts = ts_meas
        p.circle(ts.index, ts.values, color="red", fill_color="red",
                 legend="Measurement, RMSE={:.2f}".format(rmse_meas), line_width=2)
        # plot truth
        ts = ts_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
        # plot open-loop
        ts = ts_openloop
        p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
               legend="Open-loop, RMSE={:.2f}".format(rmse_openloop), line_width=2)
        # Save
        save(p)


# ------------------------------------------------------------ #
# Plot innovation (meas - y_est_before_update)
# ------------------------------------------------------------ #
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm1_openloop.loc[da_sm1_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue

        # Calculate mean y_est_before_update
        times = dict_ens_da_sm1['ens1']['time']
        data = np.empty([len(times), cfg['EnKF']['N']])
        columns = []
        for i in range(cfg['EnKF']['N']):
            data[:,i] = dict_ens_da_sm1['ens{}'.format(i+1)].loc[:, lt, lg].to_series().values
            columns.append('ens{}'.format(i+1))
        df_y_est_ens = pd.DataFrame(data, index=times, columns=columns)  # put y_est for all ensembles into a df
        y_est = np.mean(data, axis=1)
        s_y_est = pd.Series(y_est, index=times)

        # Put together with measurement
        s_meas = da_meas.loc[:, lt, lg, 0].to_series()
        df = pd.concat([s_y_est, s_meas], axis=1, keys=['y_est', 'meas']).dropna()
        df['innovation'] = df['meas'] - df['y_est']
        
        # Calculate statistics
        # --- Mean --- #
        innov_mean = df['innovation'].mean()
        # --- Normalized variance --- #
        # extract meas time points only
        df_y_est_ens = pd.concat([df_y_est_ens, s_meas], axis=1).dropna()
        df_y_est_ens = df_y_est_ens.drop(labels=0, axis=1)
        # for each time point, Pyy = cov(y, y.transpose); divided by (N-1)
        Pyy = np.cov(df_y_est_ens.values).diagonal()  # [n_times], one value for each time point
        # calculate normalized innovation time series
        df['innov_normalized'] = df['innovation'] / np.sqrt(Pyy + cfg['EnKF']['R'])
        innov_var_norm = df['innov_normalized'].var()
        
        # Plot innovation - regular plots
        fig = plt.figure(figsize=(12, 6))
        df['innovation'].plot(color='g', style='-',
                              label='Innovation (meas - y_est_before_update)\n'
                                    'mean={:.2f} var_norm={:.2f}'.format(
                                            innov_mean, innov_var_norm),
                              legend=True)
        plt.xlabel('Time')
        plt.ylabel('Innovation (mm)')
        plt.title('Innovation, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        fig.savefig(os.path.join(dirs['plots'], 'innov_{}_{}.png'.format(lt, lg)),
                    format='png')

        # Plot innovation autocorrolation (ACF)
        fig = plt.figure(figsize=(12, 6))
        pd.tools.plotting.autocorrelation_plot(df['innovation'])
        plt.xlabel('Lag (day)')
        plt.title('Innovation ACF, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        fig.savefig(os.path.join(dirs['plots'], 'innov_acf_{}_{}.png'.format(lt, lg)),
                    format='png')

        # Plot innovation - interactive
        output_file(os.path.join(dirs['plots'], 'innov_{}_{}.html'.format(lt, lg)))
        p = figure(title='Innovation, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="Innovation (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        p.line(df.index, df['innovation'].values, color="blue", line_dash="solid",
               legend="Innovation (meas - y_est_before_update)\n"
                      "mean={:.2f} var_norm={:.2f}".format(
                            innov_mean, innov_var_norm),
               line_width=2)
        save(p)


# ------------------------------------------------------------ #
# Plot results - second-layer soil moisture time series
# ------------------------------------------------------------ #
# Extract second-layer soil moisture from all datasets
da_sm2_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=1)
da_sm2_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_SOIL_MOIST'].sel(nlayer=1)
da_sm2_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=1)
dict_ens_da_sm2 = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
    dict_ens_da_sm2[ens_name] = dict_ens_ds[ens_name]['OUT_SOIL_MOIST'].sel(nlayer=1)
# Plot
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm2_openloop.loc[da_sm2_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue
        
        # --- RMSE --- #
        # extract time series
        ts_truth = da_sm2_truth.loc[:, lt, lg].to_series()
        ts_EnKF_mean = da_sm2_EnKF_ens_mean.loc[:, lt, lg].to_series()
        ts_openloop = da_sm2_openloop.loc[:, lt, lg].to_series()
        # Calculate EnKF_mean vs. truth
        df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
        rmse_EnKF_mean = rmse(df_truth_EnKF, var_true='truth', var_est='EnKF_mean')
        # Calculate open-loop vs. truth
        df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
        rmse_openloop = rmse(df_truth_openloop, var_true='truth', var_est='openloop')
        
        # ----- Regular plots ----- #
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_da_sm2[ens_name].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ts_EnKF_mean.plot(color='b', style='-',
                                      label='EnKF ens. mean, RMSE={:.2f}'.format(rmse_EnKF_mean),
                                      legend=True)
        # plot truth
        ts_truth.plot(color='k', style='-', label='Truth', legend=True)
        # plot open-loop
        ts_openloop.plot(color='m', style='--', label='Open-loop, RMSE={:.2f}'.format(rmse_openloop),
                         legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Soil moiture (mm)')
        plt.title('Second-layer soil moisture, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'sm2_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # ----- Interactive version ----- #
        # Create figure
        output_file(os.path.join(dirs['plots'], 'sm2_{}_{}.html'.format(lt, lg)))
        
        p = figure(title='Second-layer soil moisture, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="Soil moiture (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend="Ensemble members"
            else:
                legend=False
            ts = dict_ens_da_sm2[ens_name].loc[:, lt, lg].to_series()
            p.line(ts.index, ts.values, color="grey", line_dash="solid", alpha=0.3, legend=legend)
        # plot EnKF post-processed ens. mean
        ts = ts_EnKF_mean
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="EnKF ens. mean, RMSE={:.2f}".format(rmse_EnKF_mean), line_width=2)
        # plot truth
        ts = ts_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
        # plot open-loop
        ts = ts_openloop
        p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
               legend="Open-loop, RMSE={:.2f}".format(rmse_openloop), line_width=2)
        # Save
        save(p)
        
 
# ------------------------------------------------------------ #
# Plot results - third-layer soil moisture time series
# ------------------------------------------------------------ #
# Extract third-layer soil moisture from all datasets
da_sm3_truth = ds_truth['OUT_SOIL_MOIST'].sel(nlayer=2)
da_sm3_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_SOIL_MOIST'].sel(nlayer=2)
da_sm3_openloop = ds_openloop['OUT_SOIL_MOIST'].sel(nlayer=2)
dict_ens_da_sm3 = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
    dict_ens_da_sm3[ens_name] = dict_ens_ds[ens_name]['OUT_SOIL_MOIST'].sel(nlayer=2)
# Plot
for lt in lat:
    for lg in lon:
        if np.isnan(da_sm3_openloop.loc[da_sm3_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue
        
        # --- RMSE --- #
        # extract time series
        ts_truth = da_sm3_truth.loc[:, lt, lg].to_series()
        ts_EnKF_mean = da_sm3_EnKF_ens_mean.loc[:, lt, lg].to_series()
        ts_openloop = da_sm3_openloop.loc[:, lt, lg].to_series()
        # Calculate EnKF_mean vs. truth
        df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
        rmse_EnKF_mean = rmse(df_truth_EnKF, var_true='truth', var_est='EnKF_mean')
        # Calculate open-loop vs. truth
        df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
        rmse_openloop = rmse(df_truth_openloop, var_true='truth', var_est='openloop')
        
        # ----- Regular plots ----- #
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_da_sm3[ens_name].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ts_EnKF_mean.plot(color='b', style='-',
                                      label='EnKF ens. mean, RMSE={:.2f}'.format(rmse_EnKF_mean),
                                      legend=True)
        # plot truth
        ts_truth.plot(color='k', style='-', label='Truth', legend=True)
        # plot open-loop
        ts_openloop.plot(color='m', style='--', label='Open-loop, RMSE={:.2f}'.format(rmse_openloop),
                         legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Soil moiture (mm)')
        plt.title('Third-layer soil moisture, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'sm3_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # ----- Interactive version ----- #
        # Create figure
        output_file(os.path.join(dirs['plots'], 'sm3_{}_{}.html'.format(lt, lg)))
        
        p = figure(title='Third-layer soil moisture, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="Soil moiture (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend="Ensemble members"
            else:
                legend=False
            ts = dict_ens_da_sm3[ens_name].loc[:, lt, lg].to_series()
            p.line(ts.index, ts.values, color="grey", line_dash="solid", alpha=0.3, legend=legend)
        # plot EnKF post-processed ens. mean
        ts = ts_EnKF_mean
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="EnKF ens. mean, RMSE={:.2f}".format(rmse_EnKF_mean), line_width=2)
        # plot truth
        ts = ts_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
        # plot open-loop
        ts = ts_openloop
        p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
               legend="Open-loop, RMSE={:.2f}".format(rmse_openloop), line_width=2)
        # Save
        save(p)
        

# ------------------------------------------------------------ #
# Plot results - SWE time series
# ------------------------------------------------------------ #
# Extract SWE from all datasets
da_swe_truth = ds_truth['OUT_SWE']
da_swe_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_SWE']
da_swe_openloop = ds_openloop['OUT_SWE']
dict_ens_da_swe = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
    dict_ens_da_swe[ens_name] = dict_ens_ds[ens_name]['OUT_SWE']
# Plot
for lt in lat:
    for lg in lon:
        if np.isnan(da_swe_openloop.loc[da_swe_openloop['time'][0],
                                        lt, lg].values) == True:  # if inactive cell, skip
            continue
        
        # --- RMSE --- #
        # extract time series
        ts_truth = da_swe_truth.loc[:, lt, lg].to_series()
        ts_EnKF_mean = da_swe_EnKF_ens_mean.loc[:, lt, lg].to_series()
        ts_openloop = da_swe_openloop.loc[:, lt, lg].to_series()
        # Calculate EnKF_mean vs. truth
        df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
        rmse_EnKF_mean = rmse(df_truth_EnKF, var_true='truth', var_est='EnKF_mean')
        # Calculate open-loop vs. truth
        df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
        rmse_openloop = rmse(df_truth_openloop, var_true='truth', var_est='openloop')
        
        # ----- Regular plots ----- #
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_da_swe[ens_name].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ts_EnKF_mean.plot(color='b', style='-',
                                      label='EnKF ens. mean, RMSE={:.2f}'.format(rmse_EnKF_mean),
                                      legend=True)
        # plot truth
        ts_truth.plot(color='k', style='-', label='Truth', legend=True)
        # plot open-loop
        ts_openloop.plot(color='m', style='--', label='Open-loop, RMSE={:.2f}'.format(rmse_openloop),
                         legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('SWE (mm)')
        plt.title('SWE, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'swe_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # ----- Interactive version ----- #
        # Create figure
        output_file(os.path.join(dirs['plots'], 'swe_{}_{}.html'.format(lt, lg)))
        
        p = figure(title='SWE, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="SWE (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend="Ensemble members"
            else:
                legend=False
            ts = dict_ens_da_swe[ens_name].loc[:, lt, lg].to_series()
            p.line(ts.index, ts.values, color="grey", line_dash="solid", alpha=0.3, legend=legend)
        # plot EnKF post-processed ens. mean
        ts = ts_EnKF_mean
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="EnKF ens. mean, RMSE={:.2f}".format(rmse_EnKF_mean), line_width=2)
        # plot truth
        ts = ts_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
        # plot open-loop
        ts = ts_openloop
        p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
               legend="Open-loop, RMSE={:.2f}".format(rmse_openloop), line_width=2)
        # Save
        save(p)


# ------------------------------------------------------------ #
# Plot results - surface runoff time series
# ------------------------------------------------------------ #
# Extract SWE from all datasets
da_runoff_truth = ds_truth['OUT_RUNOFF']
da_runoff_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_RUNOFF']
da_runoff_openloop = ds_openloop['OUT_RUNOFF']
dict_ens_da_runoff = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
    dict_ens_da_runoff[ens_name] = dict_ens_ds[ens_name]['OUT_RUNOFF']
# Plot
for lt in lat:
    for lg in lon:
        if np.isnan(da_runoff_openloop.loc[da_runoff_openloop['time'][0],
                                           lt, lg].values) == True:  # if inactive cell, skip
            continue
        
        # --- RMSE --- #
        # extract time series
        ts_truth = da_runoff_truth.loc[:, lt, lg].to_series()
        ts_EnKF_mean = da_runoff_EnKF_ens_mean.loc[:, lt, lg].to_series()
        ts_openloop = da_runoff_openloop.loc[:, lt, lg].to_series()
        # Calculate EnKF_mean vs. truth
        df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
        rmse_EnKF_mean = rmse(df_truth_EnKF, var_true='truth', var_est='EnKF_mean')
        # Calculate open-loop vs. truth
        df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
        rmse_openloop = rmse(df_truth_openloop, var_true='truth', var_est='openloop')
        
        # ----- Regular plots ----- #
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_da_runoff[ens_name].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ts_EnKF_mean.plot(color='b', style='-',
                                      label='EnKF ens. mean, RMSE={:.2f}'.format(rmse_EnKF_mean),
                                      legend=True)
        # plot truth
        ts_truth.plot(color='k', style='-', label='Truth', legend=True)
        # plot open-loop
        ts_openloop.plot(color='m', style='--', label='Open-loop, RMSE={:.2f}'.format(rmse_openloop),
                         legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Runoff (mm)')
        plt.title('Surface runoff, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'runoff_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # ----- Interactive version ----- #
        # Create figure
        output_file(os.path.join(dirs['plots'], 'runoff_{}_{}.html'.format(lt, lg)))
        
        p = figure(title='Surface runoff, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="Runoff (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend="Ensemble members"
            else:
                legend=False
            ts = dict_ens_da_runoff[ens_name].loc[:, lt, lg].to_series()
            p.line(ts.index, ts.values, color="grey", line_dash="solid", alpha=0.3, legend=legend)
        # plot EnKF post-processed ens. mean
        ts = ts_EnKF_mean
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="EnKF ens. mean, RMSE={:.2f}".format(rmse_EnKF_mean), line_width=2)
        # plot truth
        ts = ts_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
        # plot open-loop
        ts = ts_openloop
        p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
               legend="Open-loop, RMSE={:.2f}".format(rmse_openloop), line_width=2)
        # Save
        save(p) 


# ------------------------------------------------------------ #
# Plot results - baseflow time series
# ------------------------------------------------------------ #
# Extract baseflow from all datasets
da_baseflow_truth = ds_truth['OUT_BASEFLOW']
da_baseflow_EnKF_ens_mean = ds_EnKF_ens_mean['OUT_BASEFLOW']
da_baseflow_openloop = ds_openloop['OUT_BASEFLOW']
dict_ens_da_baseflow = {}
for i in range(cfg['EnKF']['N']):
    ens_name = 'ens{}'.format(i+1)
    dict_ens_da_baseflow[ens_name] = dict_ens_ds[ens_name]['OUT_BASEFLOW']
# Plot
for lt in lat:
    for lg in lon:
        if np.isnan(da_baseflow_openloop.loc[da_baseflow_openloop['time'][0],
                                             lt, lg].values) == True:  # if inactive cell, skip
            continue
        
        # --- RMSE --- #
        # extract time series
        ts_truth = da_baseflow_truth.loc[:, lt, lg].to_series()
        ts_EnKF_mean = da_baseflow_EnKF_ens_mean.loc[:, lt, lg].to_series()
        ts_openloop = da_baseflow_openloop.loc[:, lt, lg].to_series()
        # Calculate EnKF_mean vs. truth
        df_truth_EnKF = pd.concat([ts_truth, ts_EnKF_mean], axis=1, keys=['truth', 'EnKF_mean']).dropna()
        rmse_EnKF_mean = rmse(df_truth_EnKF, var_true='truth', var_est='EnKF_mean')
        # Calculate open-loop vs. truth
        df_truth_openloop = pd.concat([ts_truth, ts_openloop], axis=1, keys=['truth', 'openloop']).dropna()
        rmse_openloop = rmse(df_truth_openloop, var_true='truth', var_est='openloop')
        
        # ----- Regular plots ----- #
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend=True
            else:
                legend=False
            dict_ens_da_baseflow[ens_name].loc[:, lt, lg].to_series().plot(
                        color='grey', style='-', alpha=0.3, label='Ensemble members',
                        legend=legend)
        # plot EnKF post-processed ens. mean
        ts_EnKF_mean.plot(color='b', style='-',
                                      label='EnKF ens. mean, RMSE={:.2f}'.format(rmse_EnKF_mean),
                                      legend=True)
        # plot truth
        ts_truth.plot(color='k', style='-', label='Truth', legend=True)
        # plot open-loop
        ts_openloop.plot(color='m', style='--', label='Open-loop, RMSE={:.2f}'.format(rmse_openloop),
                         legend=True)
        # Make plot looks better
        plt.xlabel('Time')
        plt.ylabel('Baseflow (mm)')
        plt.title('Baseflow, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']))
        # Save figure
        fig.savefig(os.path.join(dirs['plots'], 'baseflow_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # ----- Interactive version ----- #
        # Create figure
        output_file(os.path.join(dirs['plots'], 'baseflow_{}_{}.html'.format(lt, lg)))
        
        p = figure(title='Baseflow, {}, {}, N={}'.format(lt, lg, cfg['EnKF']['N']),
                   x_axis_label="Time", y_axis_label="Baseflow (mm)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot each ensemble member
        for i in range(cfg['EnKF']['N']):
            ens_name = 'ens{}'.format(i+1)
            if i == 0:
                legend="Ensemble members"
            else:
                legend=False
            ts = dict_ens_da_baseflow[ens_name].loc[:, lt, lg].to_series()
            p.line(ts.index, ts.values, color="grey", line_dash="solid", alpha=0.3, legend=legend)
        # plot EnKF post-processed ens. mean
        ts = ts_EnKF_mean
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="EnKF ens. mean, RMSE={:.2f}".format(rmse_EnKF_mean), line_width=2)
        # plot truth
        ts = ts_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", legend="Truth", line_width=2)
        # plot open-loop
        ts = ts_openloop
        p.line(ts.index, ts.values, color="magenta", line_dash="dashed",
               legend="Open-loop, RMSE={:.2f}".format(rmse_openloop), line_width=2)
        # Save
        save(p)

