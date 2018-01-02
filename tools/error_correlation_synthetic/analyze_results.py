import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import argparse

from tonic.io import read_config, read_configobj

from da_utils import setup_output_dirs, rmse


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
                    help='Ensemble size')
args = parser.parse_args()

# Read config file
cfg = read_configobj(args.cfg)
# Correlation coefficient of perturbation
corrcoef = args.corrcoef
# Autocorrelation parameter of AR(1) for both state and forcing perturbation
phi = args.phi
# Perturbation ensemble size
N = args.N


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

# --- Outputs --- #
output_dir = os.path.join(root_dir, cfg['OUTPUT']['output_dir'])


# ===================================================== #
# Load data
# ===================================================== #
print('Loading history files...')
list_hist = []
# --- Load history files --- #
for i in range(N):
    hist_nc = os.path.join(output_dir, 'vic_output',
                           'corrcoef_{}_phi_{}'.format(corrcoef, phi), 'history', 'ens_{}'.format(i+1),
                           'history.concat.{}_{}.nc'.format(start_year, end_year))
    list_hist.append(xr.open_dataset(hist_nc))

# --- Load orig. history file --- #
ds_hist_orig = xr.open_dataset(hist_orig_nc)


# ===================================================== #
# Set up output plot directory
# ===================================================== #
output_plot_dir = setup_output_dirs(
    output_dir, mkdirs=['plots'])['plots']
output_plot_dir = setup_output_dirs(
    output_plot_dir, mkdirs=['corr_{}_phi_{}'.format(corrcoef, phi)])\
    ['corr_{}_phi_{}'.format(corrcoef, phi)]


# ===================================================== #
# Plot
# ===================================================== #
print('Plotting fast-response runoff...')
# --- Plot surface runoff ensemble --- #
# Create figure
output_file(os.path.join(output_plot_dir, 'runoff.N{}.html'.format(N)))

p = figure(title='Fast-response runoff, corr. = {}, phi = {}, N = {}'.format(corrcoef, phi, N),
           x_axis_label="Time", y_axis_label="Runoff (mm/timestep)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each ensemble member
for i, ds_hist in enumerate(list_hist):
    if i == 0:
        legend="Ensemble members"
    else:
        legend=False
    ts = ds_hist['OUT_RUNOFF'].squeeze().to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# Plot orig. runoff
ts = ds_hist_orig['OUT_RUNOFF'].squeeze().to_series()
p.line(ts.index, ts.values, color="black", line_dash="solid", alpha=1, legend="Original")
# Save
save(p)

# --- Plot baseflow ensemble --- #
print('Plotting slow-response runoff...')
# Create figure
output_file(os.path.join(output_plot_dir, 'baseflow.N{}.html'.format(N)))

p = figure(title='Slow-response runoff, corr. = {}, phi = {}, N = {}'.format(corrcoef, phi, N),
           x_axis_label="Time", y_axis_label="Runoff (mm/timestep)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each ensemble member
for i, ds_hist in enumerate(list_hist):
    if i == 0:
        legend="Ensemble members"
    else:
        legend=False
    ts = ds_hist['OUT_BASEFLOW'].squeeze().to_series()
    p.line(ts.index, ts.values, color="blue", line_dash="solid", alpha=0.3, legend=legend)
# Plot orig. runoff
ts = ds_hist_orig['OUT_BASEFLOW'].squeeze().to_series()
p.line(ts.index, ts.values, color="black", line_dash="solid", alpha=1, legend="Original")
# Save
save(p)


# ===================================================== #
# Calculate metrics
# ===================================================== #
# Prepare ensemble runoff results
list_runoff = []
list_baseflow = []
for i, ds_hist in enumerate(list_hist):
    list_runoff.append(ds_hist['OUT_RUNOFF'].squeeze().to_series())
    list_baseflow.append(ds_hist['OUT_BASEFLOW'].squeeze().to_series())
df_runoff = pd.concat(list_runoff, axis=1)
df_runoff.columns = range(1, N+1)
df_baseflow = pd.concat(list_baseflow, axis=1)
df_baseflow.columns = range(1, N+1)

# Prepare orig. runoff results
ts_runoff_orig = ds_hist_orig['OUT_RUNOFF'].squeeze().to_series()
ts_baseflow_orig = ds_hist_orig['OUT_BASEFLOW'].squeeze().to_series()

# --- Relative bias of ensemble mean --- #
bias_runoff = df_runoff.sum().mean() / ts_runoff_orig.sum()
bias_baseflow = df_baseflow.sum().mean() / ts_baseflow_orig.sum()

# --- Relative bias of ensemble median --- #
bias_runoff_median = df_runoff.sum().median() / ts_runoff_orig.sum()
bias_baseflow_median = df_baseflow.sum().median() / ts_baseflow_orig.sum()

# --- logRMSE of ensemble mean --- #
ts_runoff_mean = df_runoff.mean(axis=1)
ts_true = np.log(ts_runoff_orig+1)
rmse_runoff = rmse(ts_true, np.log(ts_runoff_mean+1))

ts_baseflow_mean = df_baseflow.mean(axis=1)
ts_true = np.log(ts_baseflow_orig+1)
rmse_baseflow = rmse(ts_true, np.log(ts_baseflow_mean+1))

# --- Normalized MIQR (normalized by orig. time series mean) --- #
# Runoff
q75 = np.percentile(df_runoff, 75, axis=1, interpolation='higher')
q25 = np.percentile(df_runoff, 25, axis=1, interpolation='lower')
MIQR_runoff = (q75 - q25).mean() / ts_runoff_orig.mean()
# Baseflow
q75 = np.percentile(df_baseflow, 75, axis=1, interpolation='higher')
q25 = np.percentile(df_baseflow, 25, axis=1, interpolation='lower')
MIQR_baseflow = (q75 - q25).mean() / ts_baseflow_orig.mean()

# --- Write results to file --- #
f = open(os.path.join(output_plot_dir, 'metrics.N{}.txt'.format(N)), 'w')
f.write('Correlation coefficient = {}, phi = {}, N = {}\n\n'.format(corrcoef, phi, N))
f.write('Relative bias of ensemble mean, fast-response runoff: {}\n'.format(bias_runoff))
f.write('Relative bias of ensemble mean, slow-response runoff: {}\n'.format(bias_baseflow))
f.write('\n')
f.write('Relative bias of ensemble median, fast-response runoff: {}\n'.format(bias_runoff_median))
f.write('Relative bias of ensemble median, slow-response runoff: {}\n'.format(bias_baseflow_median))
f.write('\n')
f.write('logRMSE of ensemble mean, fast-response runoff: {}\n'.format(rmse_runoff))
f.write('logRMSE of ensemble mean, slow-response runoff: {}\n'.format(rmse_baseflow))
f.write('\n')
f.write('Normalized MIQR, fast-response runoff: {}\n'.format(MIQR_runoff))
f.write('Normalized MIQR, slow-response runoff: {}\n'.format(MIQR_baseflow))
f.write('\n')
f.close()



