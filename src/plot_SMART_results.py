
''' This script plots SMART results, including:
        1) Innovation and lambda parameter
        2) Precipitation RMSE's

    Usage:
        $ python plot_SMART_results.py <config_file_SMART>
'''

import sys
import os
import pandas as pd
import xarray as xr
from scipy.io import loadmat
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh

from tonic.io import read_configobj

from da_utils import (load_nc_and_concat_var_years, da_2D_to_3D_from_SMART,
                      setup_output_dirs, rmse)


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
# Set up output directory
# ============================================================ #
output_dir = setup_output_dirs(
                    os.path.join(cfg['CONTROL']['root_dir'],
                                 cfg['OUTPUT']['output_basedir']),
                    mkdirs=['plots'])['plots']

output_subdir_maps = setup_output_dirs(
                            output_dir,
                            mkdirs=['maps'])['maps']
output_subdir_ts = setup_output_dirs(
                            output_dir,
                            mkdirs=['time_series'])['time_series']


# ============================================================ #
# Load data
# ============================================================ #
print('Loading data...')

# --- Corrected prec --- #
da_prec_corr = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['OUTPUT']['output_basedir'],
                                          'post_SMART',
                                          'prec_corrected.'),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_corr': 'prec_corrected'})\
                  ['prec_corr']  # [time, lat, lon]

# --- Truth prec --- #
da_prec_true = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_true_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_true': cfg['PREC']['prec_true_varname']})\
               ['prec_true']  # [time, lat, lon]

# --- Original prec --- #
da_prec_orig = load_nc_and_concat_var_years(
                    basepath=os.path.join(cfg['CONTROL']['root_dir'],
                                          cfg['PREC']['prec_orig_nc_basepath']),
                    start_year=start_year,
                    end_year=end_year,
                    dict_vars={'prec_orig': cfg['PREC']['prec_orig_varname']})\
                  ['prec_orig']  # [time, lat, lon]

# --- SMART innovation --- #
innov = loadmat(os.path.join(cfg['CONTROL']['root_dir'],
                             cfg['OUTPUT']['output_basedir'],
                             'run_SMART',
                             'innovation.mat'))['innovation']  # [time, pixel]

# --- SMART lambda parameter --- #
lambda_param = loadmat(os.path.join(
                    cfg['CONTROL']['root_dir'],
                    cfg['OUTPUT']['output_basedir'],
                    'run_SMART',
                    'lambda.mat'))['lambda']  # [pixel]

# --- Domain mask --- #
da_mask = xr.open_dataset(os.path.join(
                cfg['CONTROL']['root_dir'],
                cfg['DOMAIN']['domain_file']))['mask']


# ============================================================ #
# Process innovation and lambda param
# ============================================================ #

# --- Convert innovation to [time, lat, lon] --- #
da_innov = da_2D_to_3D_from_SMART(
                dict_array_2D={'innov': innov},
                da_mask=da_mask,
                out_time_varname='time',
                out_time_coord=pd.date_range(start_date, end_date, freq='D'))\
           ['innov']

# --- Convert lambda param to [lat, lon] --- #
da_lambda = da_2D_to_3D_from_SMART(
                dict_array_2D={'lambda': lambda_param},
                da_mask=da_mask,
                out_time_varname='time',
                out_time_coord=['0'])['lambda'].sel(time='0')


# ============================================================ #
# Plot maps
# ============================================================ #

# --- Innovation maps --- #
# (1) Mean
da_innov_mean = da_innov.mean(dim='time')
fig = plt.figure(figsize=(14, 7))
cs = da_innov_mean.plot(add_colorbar=False, cmap='bwr', vmin=-0.1, vmax=0.1)
cbar = plt.colorbar(cs, extend='both').set_label('Innovation', fontsize=20)
plt.title('Mean normalized innovation (meas - y_est_before_update)\n'
          'Avg. value: {:.2f}'.format(float(da_innov_mean.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_subdir_maps, 'innov_mean.png'), format='png')
# (2) Normalized variance
da_innov_var = da_innov.var(dim='time')
fig = plt.figure(figsize=(14, 7))
cs = da_innov_var.plot(add_colorbar=False, cmap='bwr', vmin=0, vmax=2)
cbar = plt.colorbar(cs, extend='max').set_label('Normalized variance', fontsize=20)
plt.title('Normalized innovation variance, '
          'avg. value: {:.2f}'.format(float(da_innov_var.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_subdir_maps, 'innov_var_norm.png'), format='png')

# --- Lambda param map --- #
fig = plt.figure(figsize=(14, 7))
cs = da_lambda.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=0.5)
cbar = plt.colorbar(cs, extend='max').set_label('Lambda', fontsize=20)
plt.title('SMART lambda param, '
          'avg. value: {:.2f}'.format(float(da_lambda.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_subdir_maps, 'lambda_param.png'), format='png')

# --- Prec RMSE maps --- #
lat_coord = da_prec_true['lat']
lon_coord = da_prec_true['lon']
time_coord = da_prec_true['time']

# Determine the total number of loops
nloop = len(lat_coord) * len(lon_coord)
# Reshape variables
truth = da_prec_true.values.reshape([len(time_coord), nloop])  # [time, nloop]
corr = da_prec_corr.values.reshape([len(time_coord), nloop])  # [time, nloop]
orig = da_prec_orig.values.reshape([len(time_coord), nloop])  # [time, nloop]
# Calculate RMSE for all grid cells
rmse_corr = np.array(list(map(
            lambda j: rmse(truth[:, j], corr[:, j]),
            range(nloop))))  # [nloop]
rmse_orig = np.array(list(map(
            lambda j: rmse(truth[:, j], orig[:, j]),
            range(nloop))))  # [nloop]
# Reshape RMSE's
rmse_corr = rmse_corr.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
rmse_orig = rmse_orig.reshape([len(lat_coord), len(lon_coord)])  # [lat, lon]
# Put results into da's
da_rmse_corr = xr.DataArray(rmse_corr, coords=[lat_coord, lon_coord],
                            dims=['lat', 'lon'])
da_rmse_orig = xr.DataArray(rmse_orig, coords=[lat_coord, lon_coord],
                            dims=['lat', 'lon'])

# Plot map - RMSE, orig. vs. truth
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_orig.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=2)
cbar = plt.colorbar(cs, extend='max').set_label('Prec. (mm)', fontsize=20)
plt.title('Prec., RMSE of orig. (wrt. truth) '
          'avg. value: {:.2f}'.format(float(da_rmse_orig.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_subdir_maps, 'rmse_prec_orig.png'), format='png')

# Plot map - RMSE, corr. vs. truth
fig = plt.figure(figsize=(14, 7))
cs = da_rmse_corr.plot(add_colorbar=False, cmap='viridis', vmin=0, vmax=2)
cbar = plt.colorbar(cs, extend='max').set_label('Prec. (mm)', fontsize=20)
plt.title('Prec., RMSE of SMART corrected (wrt. truth) '
          'avg. value: {:.2f}'.format(float(da_rmse_corr.mean().values)), fontsize=20)
fig.savefig(os.path.join(output_subdir_maps, 'rmse_prec_corr.png'), format='png')

# Plot map - RMSE diff (corr - orig)
fig = plt.figure(figsize=(14, 7))
cs = (da_rmse_corr - da_rmse_orig).plot(
                add_colorbar=False, cmap='bwr', vmin=-0.1, vmax=0.1)
cbar = plt.colorbar(cs, extend='both').set_label('Prec. (mm)', fontsize=20)
plt.title('Prec., RMSE diff. (SMART corrected - orig., both wrt. truth)',
          fontsize=20)
fig.savefig(os.path.join(output_subdir_maps, 'rmse_prec_diff.png'), format='png')


# ============================================================ #
# Plot time series
# ============================================================ #
lat = da_mask['lat'].values
lon = da_mask['lon'].values

count = 0
for lt in lat:
    for lg in lon:
        # if inactive cell, skip
        if da_mask.loc[lt, lg] <= 0 or np.isnan(da_mask.loc[lt, lg]) == True:
            continue

        # --- Extract time series, aggregate to daily and calculate statistics --- #
        ts_prec_truth = da_prec_true.loc[:, lt, lg].to_series()
        ts_prec_truth_daily = ts_prec_truth.resample('D').mean()
        ts_prec_corr = da_prec_corr.loc[:, lt, lg].to_series()
        ts_prec_corr_daily = ts_prec_corr.resample('D').mean()
        ts_prec_orig = da_prec_orig.loc[:, lt, lg].to_series()
        ts_prec_orig_daily = ts_prec_orig.resample('D').mean()
        # Calculate rmse
        df_daily = pd.concat([ts_prec_truth_daily, ts_prec_corr_daily, ts_prec_orig_daily],
                             axis=1,
                             keys=['truth', 'corrected', 'orig'])
        rmse_orig = rmse(df_daily['truth'], df_daily['orig'])
        rmse_corrected = rmse(df_daily['truth'], df_daily['corrected'])

        # --- Prec, interactive plot --- #
        # Create figure
        output_file(os.path.join(output_subdir_ts, 'prec.{}_{}.html'.format(lt, lg)))

        p = figure(title='Precipitation, {}, {}'.format(lt, lg),
                   x_axis_label="Time", y_axis_label="Precipitation (mm/step)",
                   x_axis_type='datetime', width=1000, height=500)
        # plot truth
        ts = ts_prec_truth
        p.line(ts.index, ts.values, color="black", line_dash="solid", alpha=0.3,
               legend="Truth (orig. prec plus perturbation), hourly", line_width=2)
        # plot truth, daily
        ts = ts_prec_truth.resample('D').mean()
        p.line(ts.index, ts.values, color="black", line_dash="solid",
               legend="Truth (orig. prec plus perturbation), daily", line_width=2)
        # plot corrected prec, daily
        ts = ts_prec_corr.resample('D').mean()
        p.line(ts.index, ts.values, color="blue", line_dash="solid",
               legend="Corrected prec (via SMART), daily\n"
                      "RMSE = {:.2f}".format(rmse_corrected),
               line_width=2)
        # plot orig. prec, daily
        ts = ts_prec_orig.resample('D').mean()
        p.line(ts.index, ts.values, color="red", line_dash="dashed",
               legend="Orig. prec (before correction), daily\n"
                      "RMSE = {:.2f}".format(rmse_orig),
               line_width=2)
        # Save
        save(p)
        
        # --- Innovation, regular plot --- #
        fig = plt.figure(figsize=(12, 6))
        ts = da_innov.loc[:, lt, lg].to_series()
        ts.plot(color='g', style='-',
                label='Normalized innovation (meas - y_est_before_update)\n'
                'mean={:.2f} var={:.2f}'.format(
                            ts.mean(), ts.var()),
                legend=True)
        plt.xlabel('Time')
        plt.ylabel('Normalized innovation')
        plt.title('Normalized innovation, {}, {}'.format(lt, lg))
        fig.savefig(os.path.join(output_subdir_ts, 'innov_{}_{}.png'.format(lt, lg)),
                    format='png')
        
        # --- Innovation, autocorrelation --- #
        fig = plt.figure(figsize=(12, 6))
        pd.tools.plotting.autocorrelation_plot(ts)
        plt.xlabel('Lag (day)')
        plt.title('Innovation ACF, {}, {}'.format(lt, lg))
        fig.savefig(os.path.join(output_subdir_ts, 'innov_acf_{}_{}.png'.format(lt, lg)),
                    format='png')

        count += 1
        if count > 21:
            break
    if count > 21:
        break

