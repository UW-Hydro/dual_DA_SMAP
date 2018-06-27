
import sys
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import multiprocessing as mp
import xarray as xr

from tonic.io import read_configobj
from analysis_utils import crps, setup_output_dirs, get_soil_depth


def calculate_crps(out_nc, ds_truth, ds_model, var, depth_sm=None, nproc=1):
    ''' A wrap funciton that calculates CRPS for all domain and save to file; if
        result file already existed, then simply read in the file.

    Parameters
    ----------
    out_nc: <str>
        RMSE result output netCDF file
    ds_truth: <xr.Dataset>
        Truth states/history
    ds_model: <xr.Dataset>
        Model states/history whose RMSE is to be assessed (wrt. truth states);
        This should be ensemble model results, with "N" as the ensemble dimension
    var: <str>
        Variable, options:
            sm1; sm2; sm3; runoff_daily_log; baseflow_daily_log; totrunoff_daily_log
        NOTE: sm's and swe are from states; runoff's are from history file
    depth_sm: <xr.DataArray>
        Thickness of soil moisture
        Only required if state_var is soil moisture
    nproc: <int>
        Number of processors for mp

    Returns
    ----------
    da_crps: <xr.DataArray>
        CRPS for the whole domain; dimension: [lat, lon]
    '''

    if not os.path.isfile(out_nc):  # if RMSE is not already calculated
        # --- Extract variables --- #
        if var == 'sm1':
            da_truth = ds_truth['SOIL_MOISTURE'].sel(nlayer=0) / depth_sm
            da_model = ds_model['SOIL_MOISTURE'].sel(nlayer=0) / depth_sm
        elif var == 'sm2':
            da_truth = ds_truth['SOIL_MOISTURE'].sel(nlayer=1) / depth_sm
            da_model = ds_model['SOIL_MOISTURE'].sel(nlayer=1) / depth_sm
        elif var == 'sm3':
            da_truth = ds_truth['SOIL_MOISTURE'].sel(nlayer=2) / depth_sm
            da_model = ds_model['SOIL_MOISTURE'].sel(nlayer=2) / depth_sm
        elif var == 'runoff_daily_log':
            da_truth = np.log(ds_truth['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum') + 1)
            da_model = np.log(ds_model['OUT_RUNOFF'] + 1)
        elif var == 'baseflow_daily_log':
            da_truth = np.log(ds_truth['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum') + 1)
            da_model = np.log(ds_model['OUT_BASEFLOW'] + 1)
        elif var == 'totrunoff_daily_log':
            da_truth = np.log(
                ds_truth['OUT_RUNOFF'].resample('1D', dim='time', how='sum') + \
                ds_truth['OUT_BASEFLOW'].resample('1D', dim='time', how='sum') +1)
            da_model = np.log(
                ds_model['OUT_RUNOFF'] + \
                ds_model['OUT_BASEFLOW'] + 1)

        # --- Calculate CRPS for the whole domain --- #
        results = {}
        pool = mp.Pool(processes=nproc)
        for lat in da_truth['lat'].values:
            for lon in da_truth['lon'].values:
                results[(lat, lon)] = pool.apply_async(
                    crps, (da_truth.sel(lat=lat, lon=lon).values,
                           da_model.sel(lat=lat, lon=lon).transpose('time', 'N').values))
        pool.close()
        pool.join()
        # --- Get return values --- #
        crps_domain = np.zeros([len(da_truth['lat']), len(da_truth['lon'])])
        crps_domain[:] = np.nan
        da_crps = xr.DataArray(
            crps_domain, coords=[da_truth['lat'], da_truth['lon']],
            dims=['lat', 'lon'])
        for i, result in results.items():
            lat = i[0]
            lon = i[1]
            da_crps.loc[lat, lon] = result.get()
        # Save CRPS to netCDF file
        ds_crps = xr.Dataset(
            {'crps': da_crps})
        ds_crps.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
    else:  # if RMSE is already calculated
        da_crps = xr.open_dataset(out_nc)['crps']

    return da_crps


# ========================================================== #
# Command line arguments
# ========================================================== #
cfg = read_configobj(sys.argv[1])
var = sys.argv[2]  # Options: sm1, sm2, sm3; runoff_daily_log; baseflow_daily_log; totrunoff_daily_log
nproc = int(sys.argv[3])


# ========================================================== #
# Parameter setting
# ========================================================== #

# --- Synthetic analysis output data directory --- #
analysis_data_dir = cfg['SYNTHETIC']['analysis_data_dir']

# --- Input directory and files --- #
# EnKF results
EnKF_result_basedir = cfg['EnKF']['EnKF_result_basedir']

# Post-process results
# post_result_basedir = cfg['POSTPROCESS']['post_result_basedir']

# Synthetic results basedir
gen_synth_basedir = cfg['EnKF']['gen_synth_basedir']
truth_dirname = cfg['EnKF']['truth_dirname']
truth_nc_filename = cfg['EnKF']['truth_nc_filename']

# Synthetic analysis results directory
synth_analysis_data_dir = cfg['EnKF']['synth_analysis_dir']

# openloop
openloop_basedir = cfg['EnKF']['openloop_basedir']

# VIC global file template (for extracting param file and snow_band)
vic_global_txt = cfg['EnKF']['vic_global_txt']

# Domain netCDF file
domain_nc = cfg['EnKF']['domain_nc']

# Time period
start_time = pd.to_datetime(cfg['EnKF']['start_time'])
end_time = pd.to_datetime(cfg['EnKF']['end_time'])

# VIC parameter netCDF file
vic_param_nc = cfg['EnKF']['vic_param_nc']

# --- Measurement times --- #
meas_times = pd.date_range(
    cfg['EnKF']['meas_start_time'],
    cfg['EnKF']['meas_end_time'],
    freq=cfg['EnKF']['freq'])

# --- Plot time period --- #
plot_start_time = pd.to_datetime(cfg['EnKF']['plot_start_time'])
plot_end_time = pd.to_datetime(cfg['EnKF']['plot_end_time'])
start_year = plot_start_time.year
end_year = plot_end_time.year

# --- others --- #
N = cfg['EnKF']['N']  # number of ensemble members
ens = cfg['EnKF']['ens']  # index of ensemble member to plot for debugging plots


# ========================================================== #
# Setup output data dir
# ========================================================== #
output_rootdir = cfg['OUTPUT']['output_dir']
output_data_dir = setup_output_dirs(
        output_rootdir,
        mkdirs=['data'])['data']


# ========================================================== #
# Load data
# ========================================================== #
print('Loading data...')

# --- Domain --- #
da_domain = xr.open_dataset(domain_nc)['mask']

# --- Truth --- #
print('\tTruth history...')
ds_truth_hist = xr.open_dataset(os.path.join(
        gen_synth_basedir, truth_dirname,
        'history', truth_nc_filename))
if var == 'sm1' or var == 'sm2' or var == 'sm3':
    print('\tTruth states...')
    truth_state_nc = os.path.join(
        gen_synth_basedir,
        truth_dirname,
        'states',
        'truth_state_cellAvg.{}_{}.nc'.format(
            meas_times[0].strftime('%Y%m%d'),
            meas_times[-1].strftime('%Y%m%d')))
    ds_truth_states = xr.open_dataset(truth_state_nc)

# --- EnKF results --- #
if var == 'sm1' or var == 'sm2' or var == 'sm3':
    print('\tEnKF updated states...')
    out_nc = os.path.join(
        EnKF_result_basedir,
        'states',
        'updated_concat',
        'updated_state_cellAvg.{}_{}.ens_concat.nc'.format(
             meas_times[0].strftime('%Y%m%d'),
             meas_times[-1].strftime('%Y%m%d')))
    if os.path.isfile(out_nc):  # If already exists
        ds_EnKF_states_allEns = xr.open_dataset(out_nc)
    else:
        list_ds_allEns = []
        for i in range(N):
            print(i+1)
            ds = xr.open_dataset(os.path.join(
                EnKF_result_basedir,
                'states',
                'updated_concat',
                'updated_state_cellAvg.{}_{}.ens{}.nc'.format(
                    meas_times[0].strftime('%Y%m%d'),
                    meas_times[-1].strftime('%Y%m%d'),
                    i+1)))
            list_ds_allEns.append(ds)
        ds_EnKF_states_allEns = xr.concat(list_ds_allEns, dim='N')
        ds_EnKF_states_allEns.to_netcdf(out_nc)
    # --- EnKF ens-mean updated states --- #
    out_nc = os.path.join(
        EnKF_result_basedir,
        'states',
        'updated_concat',
        'updated_state_cellAvg.{}_{}.ens_mean.nc'.format(
             meas_times[0].strftime('%Y%m%d'),
             meas_times[-1].strftime('%Y%m%d')))
    if os.path.isfile(out_nc):  # If already exists
        ds_EnKF_states = xr.open_dataset(out_nc)
    else:
        ds_EnKF_states_allEns = xr.concat(list_ds_allEns, dim='N')
        ds_EnKF_states = ds_EnKF_states_allEns.mean('N')
        ds_EnKF_states.to_netcdf(out_nc)

print('EnKF history, daily ...')
# --- Load and concat daily history file of all ensemble members --- #
print('Concatenating...')
out_nc = os.path.join(
    EnKF_result_basedir, 'history', 'EnKF_ensemble_concat',
    'history.daily.allEns.{}_{}.nc'.format(start_year, end_year))
if os.path.isfile(out_nc):  # If already exists
    ds_hist_daily_allEns = xr.open_dataset(out_nc)
    da_runoff_daily_allEns = ds_hist_daily_allEns['OUT_RUNOFF']
    da_baseflow_daily_allEns = ds_hist_daily_allEns['OUT_BASEFLOW']
else:
    list_da_runoff_allEns = []
    list_da_baseflow_allEns = []
    for i in range(N):
        print(i+1)
        ds = xr.open_dataset(os.path.join(
            EnKF_result_basedir, 'history', 'EnKF_ensemble_concat',
            'history.daily.ens{}.concat.{}_{}.nc'.format(i+1, start_year, end_year)))
        list_da_runoff_allEns.append(ds['OUT_RUNOFF'])
        list_da_baseflow_allEns.append(ds['OUT_BASEFLOW'])
    da_runoff_daily_allEns = xr.concat(list_da_runoff_allEns, dim='N')
    da_baseflow_daily_allEns = xr.concat(list_da_baseflow_allEns, dim='N')
    ds_hist_daily_allEns = xr.Dataset({'OUT_RUNOFF': da_runoff_daily_allEns,
                                       'OUT_BASEFLOW': da_baseflow_daily_allEns})
    ds_hist_daily_allEns.to_netcdf(out_nc)
# --- EnKF mean --- #
out_nc = os.path.join(
    EnKF_result_basedir, 'history', 'EnKF_ensemble_concat',
    'history.daily.ens_mean.concat.{}_{}.nc'.format(start_year, end_year))
if os.path.isfile(out_nc):  # If already exists
    ds_hist_daily_ensMean = xr.open_dataset(out_nc)
else:
    ds_hist_daily_ensMean = ds_hist_daily_allEns.mean(dim='N')
    ds_hist_daily_ensMean.to_netcdf(out_nc, format='NETCDF4_CLASSIC')


# ======================================================== #
# Extract shared coordinates
# ======================================================== #
lat_coord = da_domain['lat']
lon_coord = da_domain['lon']


# ======================================================== #
# Extract soil layer depths
# ======================================================== #
da_soil_depth = get_soil_depth(vic_param_nc)  # [nlayer, lat, lon]
depth_sm1 = da_soil_depth.sel(nlayer=0)  # [lat, lon]
depth_sm2 = da_soil_depth.sel(nlayer=1)  # [lat, lon]
depth_sm3 = da_soil_depth.sel(nlayer=2)  # [lat, lon]


# ======================================================== #
# Calculate CRPS
# ======================================================== #
print('Calculating CRPS...')
out_nc = os.path.join(output_data_dir, 'crps_EnKF_{}.nc'.format(var))
if var == 'sm1':
    depth_sm = depth_sm1
elif var == 'sm2':
    depth_sm = depth_sm2
elif var == 'sm3':
    depth_sm = depth_sm3
else:
    depth_sm = None

if var == 'sm1' or var == 'sm2' or var == 'sm3':
    da = calculate_crps(
        out_nc, ds_truth_states, ds_EnKF_states_allEns,
        var=var, depth_sm=depth_sm, nproc=nproc)
else:
    da = calculate_crps(
        out_nc, ds_truth_hist, ds_hist_daily_allEns,
        var=var, depth_sm=None, nproc=nproc)

