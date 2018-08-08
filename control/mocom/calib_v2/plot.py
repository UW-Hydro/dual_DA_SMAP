
import sys
import os
import xarray as xr
import subprocess
import string
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimize_utils import read_RVIC_output, read_USGS_data, kge
from tonic.io import read_configobj

# ======================================================== #
# Parse in parameters from MOCOM
# ======================================================== #
cfg = read_configobj(sys.argv[1])  # optimize cfg
stor_dir = sys.argv[2]  # .../run_ident/
soln_num = sys.argv[3]  # the 5-digit parameter solution number
out_plot_dir = sys.argv[4]

# ======================================================== #
# Parameter setting
# ======================================================== #
# --- RVIC --- #
rvic_convolve_template = cfg['RVIC']['rvic_convolve_template']
# Time lag between VIC/RVIC forcing and USGS local time [hour]
time_lag = cfg['RVIC']['time_lag']
# Site name in RVIC
site = cfg['RVIC']['site']

# --- USGS streamflow data --- #
usgs_data_txt = cfg['USGS']['usgs_data_txt']

# --- Time period for calculating statistics --- #
start_time = cfg['TIME']['start_time']
end_time = cfg['TIME']['end_time']


# ======================================================== #
# Plot and calculate objective function values
# ======================================================== #
# --- Load USGS streamflow data --- #
ts_usgs = read_USGS_data(
    usgs_data_txt, columns=[1], names=['flow'])['flow'].truncate(
        before=start_time, after=end_time)

# --- Load and process simulated flow --- #
# Load
df_routed, dict_outlet = read_RVIC_output(
    os.path.join(stor_dir, soln_num, 'rvic_output', 'hist', 'calibration.rvic.h0a.2018-01-01.nc'))
# Convert to ts
ts_routed = df_routed.loc[:, site].truncate(before=start_time, after=end_time)

# --- Calculate daily KGE --- #
kge_daily = kge(ts_routed, ts_usgs)

# --- Calculate 5-day KGE --- #
ts_routed_5D = ts_routed.resample('5D', how='sum')
ts_usgs_5D = ts_usgs.resample('5D', how='sum')
kge_5D = kge(ts_routed_5D, ts_usgs_5D)

# --- Plot time series --- #
fig = plt.figure(figsize=(20, 6))
# Routed
(ts_routed / 1000).plot(
    color='magenta', style='-',
    label='Open-loop, KGE_daily={:.2f} KGE_5D={:.2f}'.format(kge_daily, kge_5D))
# USGS
(ts_usgs / 1000).plot(color='black', label='USGS')
# Make plot better
plt.ylabel('Streamflow (thousand cfs)', fontsize=16)
plt.legend(fontsize=16)
plt.title(site, fontsize=16)
# Save to file
fig.savefig(os.path.join(out_plot_dir, '{}.{}.flow_daily.png'.format(site, soln_num)),
            format='png', bbox_inches='tight', pad_inches=0)



