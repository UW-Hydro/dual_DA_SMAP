
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import sys
import datetime as dt
import pandas as pd
import os
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import properscoring as ps

from tonic.io import read_config, read_configobj
from plot_utils import (read_USGS_data, read_RVIC_output,
                        kge, nse, rmse, nensk,
                        calc_alpha_reliability, get_z_values_timeseries,
                        calc_kesi)


# In[2]:


# ===================================================== #
# Read cfg
# ===================================================== #
cfg = read_configobj(sys.argv[1])

# In[3]:


# ===================================================== #
# Parameters
# ===================================================== #
# --- USGS data --- #
# Site info file; needs to have columns "short_name" and "site_no"
site_info_csv = cfg['USGS']['site_info_csv']
# Directory of USGS data
usgs_data_dir = cfg['USGS']['usgs_data_dir']

# --- Routed --- #
# Openloop nc
openloop_nc = cfg['ROUTE']['openloop_nc']

# Ensemble size
N = cfg['ROUTE']['N']
# Ensemble nc; "{}" will be replaced by ensemble index
ensemble_basenc = cfg['ROUTE']['ensemble_basenc']

# Time lag of routed data with local time [hours];
# Example: if local time is UTC-6 (ArkRed) and routed data is in UTC, then time_lag = 6
time_lag = cfg['ROUTE']['time_lag']

# --- Time --- #
start_time = pd.to_datetime(cfg['ROUTE']['start_time'])
end_time = pd.to_datetime(cfg['ROUTE']['end_time'])
start_year = start_time.year
end_year = end_time.year

# --- Domain file ("mask" and "area" will be used) --- #
domain_nc = cfg['DOMAIN']['domain_nc']

# --- RVIC output param remap nc files --- #
# "{}" will be replaced by site name
rvic_subbasin_nc = cfg['DOMAIN']['rvic_subbasin_nc']

# --- VIC forcing file basedir ("YYYY.nc" will be appended) --- #
force_basedir = cfg['BASEFLOW_FRAC']['force_basedir']

# --- VIC opnloop output history nc (this is to calculate baseflow fraction) --- #
vic_openloop_hist_nc = cfg['BASEFLOW_FRAC']['vic_openloop_hist_nc']

# --- Output --- #
output_dir = cfg['OUTPUT']['output_dir']


# In[4]:


# ===================================================== #
# Load USGS data
# ===================================================== #
# --- Load site info --- #
df_site_info = pd.read_csv(site_info_csv, dtype={'site_no': str})
dict_sites = {}  # {site: site_no}
for i in df_site_info.index:
    site = df_site_info.loc[i, 'short_name']
    site_no = df_site_info.loc[i, 'site_no']
    dict_sites[site] = site_no
    
# --- Load USGS streamflow data --- #
dict_flow_usgs = {}  # {site: ts}
for site in dict_sites.keys():
    print('Loading USGS data for site {}...'.format(site))
    site_no = dict_sites[site]
    filename = os.path.join(usgs_data_dir, '{}.txt'.format(site_no))
    ts_flow = read_USGS_data(filename, columns=[1], names=['flow'])['flow']
    dict_flow_usgs[site] = ts_flow.truncate(before=start_time, after=end_time)

# --- Get USGS drainage area (mi2) --- #
dict_usgs_drainage_area = {}  # {site: area}
for i in df_site_info.index:
    site = df_site_info.loc[i, 'short_name']
    drainage_area = df_site_info.loc[i, 'drain_area_va']
    dict_usgs_drainage_area[site] = drainage_area * 1.60934 * 1.60934  # convert [mi2] to [km2]


# In[5]:


# ===================================================== #
# Load and process routed data
# ===================================================== #
# --- Load openloop data --- #
df_openloop, dict_outlet = read_RVIC_output(openloop_nc)

# --- Load ensemble data --- #
list_da_ensemble = []
for i in range(N):
    filename = ensemble_basenc.format(i+1)
    df, dict_outlet = read_RVIC_output(filename)
    da = xr.DataArray(df, dims=['time', 'site'])
    list_da_ensemble.append(da)
# Concat all ensemble members
da_ensemble = xr.concat(list_da_ensemble, dim='N')

# --- Shift all routed data data to local time --- #
df_openloop.index = df_openloop.index - pd.DateOffset(hours=time_lag)
da_ensemble['time'] = pd.to_datetime(da_ensemble['time'].values) - pd.DateOffset(hours=time_lag)

# --- Average all routed data to daily (of local time) --- #
df_openloop_daily = df_openloop.resample('1D', how='mean')
da_ensemble_daily = da_ensemble.resample('1D', dim='time', how='mean')

# --- Calculate ensemble median --- #
da_ensMean_daily = da_ensemble_daily.median(dim='N')


# In[6]:


# ===================================================== #
# Load basin information
# ===================================================== #
ds_domain = xr.open_dataset(domain_nc)
da_area = ds_domain['area']
da_domain = ds_domain['mask']

# --- Basin domain --- #
dict_da_frac = {}
for site in dict_sites.keys():
    da_frac = xr.open_dataset(rvic_subbasin_nc.format(site))['fraction']
    dict_da_frac[site] = da_frac

# --- Basin area --- #
dict_basin_area = {}
for site in dict_sites.keys():
    basin_area = float(da_area.where(dict_da_frac[site]>0).sum())  # [m2]
    basin_area = basin_area / 1000 / 1000  # convert to [km2]
    dict_basin_area[site] = basin_area
    print(site, basin_area)


# In[12]:


# ===================================================== #
# Plot
# ===================================================== #
# Plot both 2016 summer and 2017 summer
for site in dict_sites.keys():
    # --- Some calculation --- #
    ts_usgs = dict_flow_usgs[site]
    ts_openloop = df_openloop_daily.loc[:, site]
    ts_usgs = ts_usgs[ts_openloop.index].dropna()
    ts_openloop = ts_openloop[ts_usgs.index].dropna()
    ts_ensMean = da_ensMean_daily.sel(site=site).to_series()[ts_usgs.index].dropna()
    ensemble_daily = da_ensemble_daily.sel(
        site=site, time=ts_usgs.index).transpose('time', 'N').values
    # KGE daily
    kge_openloop = kge(ts_openloop, ts_usgs)
    kge_ensMean = kge(ts_ensMean, ts_usgs)
    kge_improv = kge_ensMean - kge_openloop
    # PER(RMSE)
    rmse_openloop = rmse(ts_openloop, ts_usgs)
    rmse_ensMean = rmse(ts_ensMean, ts_usgs)
    per_RMSE = (1 - rmse_ensMean / rmse_openloop) * 100
    # Normalized ensemble skill (NENSK)
    nensk_ens = nensk(ts_usgs, ensemble_daily)
    # alpha reliability
    z_alltimes = get_z_values_timeseries(
        np.rollaxis(ensemble_daily, 0, 2), ts_usgs.values)
    alpha = calc_alpha_reliability(z_alltimes)
    kesi = calc_kesi(z_alltimes)
    
    # --- Plot regular plot - both 2016 and 2017 summer --- #
    for year in [2016, 2017]:
        fig = plt.figure(figsize=(15, 6))
        ax = plt.axes()
        plot_start_time = '{}-03-01'.format(year)
        plot_end_time = '{}-09-30'.format(year)
        # Ensemble
        for i in range(N):
            ts = da_ensemble_daily.sel(N=i, site=site).to_series() / 1000
            ts.truncate(before=plot_start_time, after=plot_end_time).plot(
                color='blue', alpha=0.05)
        # Ensemble mean
        (ts_ensMean / 1000).truncate(before=plot_start_time, after=plot_end_time).plot(
                color='blue', linewidth=2)
        # USGS
        (ts_usgs / 1000).truncate(
            before=plot_start_time, after=plot_end_time).plot(
            style='-', color='black', linewidth=2)
        # Openloop
        (ts_openloop / 1000).truncate(
            before=plot_start_time, after=plot_end_time).plot(
            color='magenta', style='--', linewidth=2)
        # Add text
        plt.text(0.7, 0.96,
                 "all-time PER(RMSE) = {:.1f}%\n".format(per_RMSE),
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, fontsize=16)
        plt.text(0.7, 0.90,
                 ("all-time KGE_improve = {:.2f}\n"
                  "         (open-loop KGE = {:.2f})\n").format(
                     kge_improv, kge_openloop),
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, fontsize=16)
        plt.text(0.7, 0.72,
                 "all-time NENSK = {:.2f}\n".format(nensk_ens),
                 horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes, fontsize=16)
#        plt.text(0.7, 0.66,
#                 r"all-time $\xi$ = {:.2f}".format(kesi),
#                 horizontalalignment='left',
#                 verticalalignment='top', transform=ax.transAxes, fontsize=16)
#        plt.text(0.7, 0.60,
#                 r"all-time $\alpha$ = {:.2f}".format(alpha),
#                 horizontalalignment='left',
#                 verticalalignment='top', transform=ax.transAxes, fontsize=16)

        # Make plot better
        plt.ylabel('Streamflow (thousand cfs)', fontsize=24)
        plt.xlabel("", fontsize=24)
        plt.title(site, fontsize=24)
        for t in ax.get_xticklabels():
            t.set_fontsize(20)
        for t in ax.get_yticklabels():
            t.set_fontsize(20)
        # Save to file
        fig.savefig(
            os.path.join(output_dir,
                         'flow_daily_for_paper2.{}.{}.png'.format(
                             site, year)),
            format='png', bbox_inches='tight', pad_inches=0)

