
import matplotlib
matplotlib.use('Agg')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh

# ================================================= #
# Parameter setting
# ================================================= #
# lat lon
lat = 31.1875
lon = -92.6875

# VIC output paths
output_history_dir = 'output/depth2_0.4/history'
state_pert_time = pd.to_datetime('1980-03-01-00-00')
openloop_hist_nc = '/civil/hydro/ymao/data_assim/output/vic/test.31.1875_-92.6875/openloop/depth2_0.4/history/history.openloop.1980-01-01-00000.nc'

# Plotting time period
start_plot_time = pd.to_datetime('1980-01-01-00-00')
end_plot_time = pd.to_datetime('1989-12-31-21-00')

# Set colorscheme
cmap = matplotlib.cm.get_cmap('Spectral')

# Perturbation amount
sm1_pert = np.array([-30, -10, -3, -1, -0.3, -0.1,
                     0.1, 0.3, 1, 3, 10, 30])
sm2_pert = np.array([-300, -100, -30, -10, -3, -1, -0.3, -0.1,
                     0.1, 0.3, 1, 3, 10, 30, 100, 300])
sm3_pert = np.array([-300, -100, -30, -10, -3, -1, -0.3, -0.1,
                     0.1, 0.3, 1, 3, 10, 30, 100, 300])

# Output plot base directory
output_basedir = 'output/depth2_0.4/plots'


# ================================================= #
# Setup output directory
# ================================================= #
output_dir = os.path.join(output_basedir,
                       'pert.{}'.format(state_pert_time.strftime('%Y%m%d')))
os.makedirs(output_dir, exist_ok=True)


# ================================================= #
# Load and process openloop
# ================================================= #
da_sm_openloop = xr.open_dataset(openloop_hist_nc)['OUT_SOIL_MOIST'].sel(
    lat=lat, lon=lon,
    time=slice(start_plot_time, end_plot_time))  # [time, nlayer]
da_runoff_openloop = xr.open_dataset(openloop_hist_nc)['OUT_RUNOFF'].sel(
    lat=lat, lon=lon,
    time=slice(start_plot_time, end_plot_time))  # surface runoff: [time]

# ================================================= #
# Plot - perturb sm1
# ================================================= #
print('Plotting perturb sm1...')
# --- Load VIC output data --- #
print('\tLoad data...')
list_da_sm = []  # [time, nlayer]
list_da_runoff = []  # [time]
for pert in sm1_pert:
    ncfile = os.path.join(
        output_history_dir,
        'history.perturbed_init_state.{}_{}_{}.{}-{:05d}.nc'.format(
            pert, 0.0, 0.0,
            state_pert_time.strftime('%Y-%m-%d'),
            state_pert_time.hour*3600+state_pert_time.second))
    ds = xr.open_dataset(ncfile)
    list_da_sm.append(ds['OUT_SOIL_MOIST'].sel(
        lat=lat, lon=lon,
        time=slice(start_plot_time, end_plot_time)))
    list_da_runoff.append(ds['OUT_RUNOFF'].sel(
        lat=lat, lon=lon,
        time=slice(start_plot_time, end_plot_time)))

# --- Set plotting color --- #
ncolors = len(sm1_pert)
colors = []
for i in range(ncolors):
    colors.append(cmap(1/(ncolors-1)*i))

# --- Plot - sm1 --- #
print('\tsm1...')
ts_openloop = da_sm_openloop.sel(nlayer=0).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=0).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm1 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm1, sm1 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm1_pert.sm1.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm1_pert.sm1.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, sm1 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm1 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - sm2 --- #
print('\tsm2...')
ts_openloop = da_sm_openloop.sel(nlayer=1).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=1).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm1 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm2, sm1 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm1_pert.sm2.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm1_pert.sm2.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, sm1 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm1 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - sm3 --- #
print('\tsm3...')
ts_openloop = da_sm_openloop.sel(nlayer=2).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=2).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm1 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm3, sm1 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm1_pert.sm3.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm1_pert.sm3.html'.format(lat, lon)))

p = figure(title='Bottom-layer soil moisture, sm1 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm1 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - surface runoff --- #
print('\tsurface runoff...')
ts_openloop = da_runoff_openloop.to_series()
list_ts_runoff = []
for da in list_da_runoff:
    list_ts_runoff.append(da.to_series())
### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm1_pert.runoff.html'.format(lat, lon)))
p = figure(title='Surface runoff, sm1 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Runoff (mm/step)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm1_pert):
    ts = list_ts_runoff[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm1 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# ================================================= #
# Plot - perturb sm2
# ================================================= #
print('Plotting perturb sm2...')
# --- Load VIC output data --- #
print('\tLoad data...')
list_da_sm = []  # [time, nlayer]
list_da_runoff = []  # [time]
for pert in sm2_pert:
    ncfile = os.path.join(
        output_history_dir,
        'history.perturbed_init_state.{}_{}_{}.{}-{:05d}.nc'.format(
            0.0, pert, 0.0,
            state_pert_time.strftime('%Y-%m-%d'),
            state_pert_time.hour*3600+state_pert_time.second))
    ds = xr.open_dataset(ncfile)
    list_da_sm.append(ds['OUT_SOIL_MOIST'].sel(
        lat=lat, lon=lon,
        time=slice(start_plot_time, end_plot_time)))
    list_da_runoff.append(ds['OUT_RUNOFF'].sel(
        lat=lat, lon=lon,
        time=slice(start_plot_time, end_plot_time)))

# --- Set plotting color --- #
ncolors = len(sm2_pert)
colors = []
for i in range(ncolors):
    colors.append(cmap(1/(ncolors-1)*i))

# --- Plot - sm1 --- #
print('\tsm1...')
ts_openloop = da_sm_openloop.sel(nlayer=0).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=0).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm2 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm1, sm2 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm2_pert.sm1.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm2_pert.sm1.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, sm2 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm2 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - sm2 --- #
print('\tsm2...')
ts_openloop = da_sm_openloop.sel(nlayer=1).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=1).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm2 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm2, sm2 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm2_pert.sm2.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm2_pert.sm2.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, sm2 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm2 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - sm3 --- #
print('\tsm3...')
ts_openloop = da_sm_openloop.sel(nlayer=2).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=2).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm2 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm3, sm2 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm2_pert.sm3.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm2_pert.sm3.html'.format(lat, lon)))

p = figure(title='Bottom-layer soil moisture, sm2 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm2 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - surface runoff --- #
print('\tsurface runoff...')
ts_openloop = da_runoff_openloop.to_series()
list_ts_runoff = []
for da in list_da_runoff:
    list_ts_runoff.append(da.to_series())
### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm2_pert.runoff.html'.format(lat, lon)))
p = figure(title='Surface runoff, sm2 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Runoff (mm/step)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm2_pert):
    ts = list_ts_runoff[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm2 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# ================================================= #
# Plot - perturb sm3
# ================================================= #
print('Plotting perturb sm3...')
# --- Load VIC output data --- #
print('\tLoad data...')
list_da_sm = []  # [time, nlayer]
list_da_runoff = []  # [time]
for pert in sm3_pert:
    ncfile = os.path.join(
        output_history_dir,
        'history.perturbed_init_state.{}_{}_{}.{}-{:05d}.nc'.format(
            0.0, 0.0, pert,
            state_pert_time.strftime('%Y-%m-%d'),
            state_pert_time.hour*3600+state_pert_time.second))
    ds = xr.open_dataset(ncfile)
    list_da_sm.append(ds['OUT_SOIL_MOIST'].sel(
        lat=lat, lon=lon,
        time=slice(start_plot_time, end_plot_time)))
    list_da_runoff.append(ds['OUT_RUNOFF'].sel(
        lat=lat, lon=lon,
        time=slice(start_plot_time, end_plot_time)))

# --- Set plotting color --- #
ncolors = len(sm3_pert)
colors = []
for i in range(ncolors):
    colors.append(cmap(1/(ncolors-1)*i))

# --- Plot - sm1 --- #
print('\tsm1...')
ts_openloop = da_sm_openloop.sel(nlayer=0).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=0).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm3_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm3 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm1, sm3 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm3_pert.sm1.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm3_pert.sm1.html'.format(lat, lon)))

p = figure(title='Top-layer soil moisture, sm3 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm3_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm3 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - sm2 --- #
print('\tsm2...')
ts_openloop = da_sm_openloop.sel(nlayer=1).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=1).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm3_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm3 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm2, sm3 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm3_pert.sm2.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm3_pert.sm2.html'.format(lat, lon)))

p = figure(title='Middle-layer soil moisture, sm3 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm3_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm3 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

# --- Plot - sm3 --- #
print('\tsm3...')
ts_openloop = da_sm_openloop.sel(nlayer=2).to_series()
list_ts_sm = []
for da in list_da_sm:
    list_ts_sm.append(da.sel(nlayer=2).to_series())

### Regular plot ###
fig = plt.figure(figsize=(12, 6))
# Plot each of the perturbed history
for i, pert in enumerate(sm3_pert):
    list_ts_sm[i].plot(
        color=colors[i], legend=True, label='sm3 pert. {} mm'.format(pert))
# Plot openloop
ts_openloop.plot(
    color='black', legend=True, label='Openloop', lw=2)
# Add labels
plt.legend()
plt.xlabel('Time', fontsize=16)
plt.ylabel('Soil moisture (mm)', fontsize=16)
plt.title('sm3, sm3 perturbed at {}, {}, {}'.format(
                state_pert_time.strftime('%Y-%m-%d'), lat, lon),
          fontsize=16)
fig.savefig(os.path.join(output_dir, '{}_{}.sm3_pert.sm3.png'.format(lat, lon)),
            format='png')

### Interactive plot ###
# Create figure
output_file(os.path.join(output_dir, '{}_{}.sm3_pert.sm3.html'.format(lat, lon)))

p = figure(title='Bottom-layer soil moisture, sm3 perturbed at {}, {}, {}'.format(
                lat, lon, state_pert_time.strftime('%Y-%m-%d')),
           x_axis_label="Time", y_axis_label="Soil moiture (mm)",
           x_axis_type='datetime', width=1000, height=500)
# Plot each of the perturbed history
for i, pert in enumerate(sm3_pert):
    ts = list_ts_sm[i]
    p.line(ts.index, ts.values, color=matplotlib.colors.rgb2hex(colors[i]),
           line_dash="solid",
           legend="sm3 pert. {} mm".format(pert), line_width=2)
# plot open-loop
ts = ts_openloop
p.line(ts.index, ts.values, color="black", line_dash="solid",
       legend="Openloop", line_width=2)
# Save
save(p)

