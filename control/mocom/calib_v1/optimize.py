
import sys
import os
import xarray as xr
import subprocess
import string
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from optimize_utils import read_RVIC_output, read_USGS_data, kge
from tonic.io import read_configobj

# ======================================================== #
# Parse in parameters from MOCOM
# ======================================================== #
cfg = read_configobj(sys.argv[1])

infilt = float(sys.argv[2])
d1 = float(sys.argv[3])
d2 = float(sys.argv[4])
d3 = float(sys.argv[5])
expt = float(sys.argv[6])
Ksat = float(sys.argv[7])

statsfile = sys.argv[8]
stor_dir = sys.argv[9]

# ======================================================== #
# Parameter setting
# ======================================================== #
# --- VIC --- #
vic_exe = cfg['VIC']['vic_exe']
mpi_exe = cfg['VIC']['mpi_exe']
mpi_proc = cfg['VIC']['mpi_proc']
vic_param_nc_template = cfg['VIC']['vic_param_nc_template']
# global template for running 2015-2017 as spinup of new parameters
vic_global_template_spinup = cfg['VIC']['vic_global_template_spinup']
# global template for running 2015-2017 for calibration statistics
vic_global_template_calib = cfg['VIC']['vic_global_template_calib']

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
# Make stor_dir
# ======================================================== #
if not os.path.exists(stor_dir):
    os.makedirs(stor_dir)

# ======================================================== #
# Run VIC and RVIC with input parameters
# ======================================================== #
# --- (1) Replace new parameters in the param nc --- #
ds_param = xr.open_dataset(vic_param_nc_template)
ds_param['infilt'][:] = infilt
ds_param['Ds'][:] = d1
ds_param['Dsmax'][:] = d2
ds_param['Ws'][:] = d3
ds_param['expt'][:] = expt
ds_param['Ksat'][:] = Ksat
ds_param.to_netcdf(os.path.join(stor_dir, 'param.nc'), format='NETCDF4_CLASSIC')
# --- Prepare global file --- #
# Spinup period
with open(vic_global_template_spinup, 'r') as global_file:
    global_param = global_file.read()
s = string.Template(global_param)
outdir_spinup = os.path.join(stor_dir, 'vic_output_spinup')
if not os.path.exists(outdir_spinup):
    os.makedirs(outdir_spinup)
global_param = s.safe_substitute(
    state_dir=outdir_spinup,
    param_nc=os.path.join(stor_dir, 'param.nc'),
    result_dir=outdir_spinup)
with open(os.path.join(stor_dir, 'global.spinup_2015_2017.txt'), mode='w') as f:
    for line in global_param:
        f.write(line)
# Calibration period
with open(vic_global_template_calib, 'r') as global_file:
    global_param = global_file.read()
s = string.Template(global_param)
outdir_calib = os.path.join(stor_dir, 'vic_output_calib')
if not os.path.exists(outdir_calib):
    os.makedirs(outdir_calib)
global_param = s.safe_substitute(
    init_state=os.path.join(outdir_spinup, 'state.20180101_00000.nc'),
    param_nc=os.path.join(stor_dir, 'param.nc'),
    result_dir=outdir_calib)
with open(os.path.join(stor_dir, 'global.calib_2015_2017.txt'), mode='w') as f:
    for line in global_param:
        f.write(line)

# --- (2) Run VIC --- #
# Spinup period
proc = subprocess.Popen(
    "{} -np {} {} -g {}".format(mpi_exe, mpi_proc, vic_exe,
                                os.path.join(stor_dir, 'global.spinup_2015_2017.txt')),
    shell=True,
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE)
retvals = proc.communicate()
returncode = proc.returncode
if returncode != 0:
    stdout = retvals[0]
    stderr = retvals[1]
    print(stderr)
    raise ValueError('VIC running error!')
# Calibration period
proc = subprocess.Popen(
    "{} -np {} {} -g {}".format(mpi_exe, mpi_proc, vic_exe,
                                os.path.join(stor_dir, 'global.calib_2015_2017.txt')),
    shell=True,
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE)
retvals = proc.communicate()
returncode = proc.returncode
if returncode != 0:
    stdout = retvals[0]
    stderr = retvals[1]
    print(stderr)
    raise ValueError('VIC running error!')

# --- (3) Shift VIC output to local time; then aggregate to daily --- #
# Load in original VIC output
ds_vic_result = xr.open_dataset(os.path.join(outdir_calib, 'fluxes.2015-01-01-00000.nc'))
# Shift time
times_orig = pd.to_datetime(ds_vic_result['time'].values)
times_new = times_orig - pd.DateOffset(hours=time_lag)
ds_vic_result['time'] = times_new
ds_vic_result = ds_vic_result.sel(time=slice('2015-01-01', '2018-01-01'))
# Aggregate to daily
ds_vic_daily = ds_vic_result.resample(dim='time', freq='1D', how='sum')
# Save to file
ds_vic_daily['OUT_RUNOFF'].attrs = ds_vic_result['OUT_RUNOFF'].attrs
ds_vic_daily['OUT_BASEFLOW'].attrs = ds_vic_result['OUT_BASEFLOW'].attrs
ds_vic_daily.to_netcdf(os.path.join(outdir_calib,
                            'fluxes.2015-01-01-00000.shifted_daily.nc'),
                       format='NETCDF4_CLASSIC')

# --- (4) Run RVIC (daily) --- #
# --- NOTE: RVIC is run for 2015-01-01 to 2017-12-30, but only 2015-03-31 to 2017-12-30
# --- will be evaluated, while the first 3 months are treated as spinup for RVIC
# Prepare control file
with open(rvic_convolve_template, 'r') as global_file:
    global_param = global_file.read()
s = string.Template(global_param)
outdir_rvic = os.path.join(stor_dir, 'rvic_output')
if not os.path.exists(outdir_rvic):
    os.makedirs(outdir_rvic)
global_param = s.safe_substitute(
    output_dir=outdir_rvic,
    vic_output_dir=outdir_calib)
with open(os.path.join(stor_dir, 'rvic.convolve.txt'), mode='w') as f:
    for line in global_param:
        f.write(line)
# Run RVIC
proc = subprocess.Popen(
    "rvic convolution {}".format(
        os.path.join(stor_dir, 'rvic.convolve.txt')),
    shell=True,
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE)
retvals = proc.communicate()
returncode = proc.returncode
if returncode != 0:
    stdout = retvals[0]
    stderr = retvals[1]
    print(stderr)
    raise ValueError('RVIC running error!')

# ======================================================== #
# Calculate objective function values
# ======================================================== #
# --- Load USGS streamflow data --- #
ts_usgs = read_USGS_data(
    usgs_data_txt, columns=[1], names=['flow'])['flow'].truncate(
        before=start_time, after=end_time)

# --- Load and process simulated flow --- #
# Load
df_routed, dict_outlet = read_RVIC_output(
    os.path.join(outdir_rvic, 'hist', 'calibration.rvic.h0a.2018-01-01.nc'))
# Convert to ts
ts_routed = df_routed.loc[:, site].truncate(before=start_time, after=end_time)

# --- Calculate daily KGE --- #
kge_daily = kge(ts_routed, ts_usgs)

# --- Calculate 5-day KGE --- #
ts_routed_5D = ts_routed.resample('5D', how='sum')
ts_usgs_5D = ts_usgs.resample('5D', how='sum')
kge_5D = kge(ts_routed_5D, ts_usgs_5D)

# ======================================================== #
# Save objective function values to statsfile (to minimize)
# ======================================================== #
# Save objective function to statsfile
dirname = os.path.dirname(statsfile)
if not os.path.exists(dirname):
    os.makedirs(dirname)
with open(statsfile, 'w') as f:
    f.write('{:.8f} {:.8f}'.format(-kge_daily, -kge_5D))

# ======================================================== #
# Save additional info to stor_dir
# ======================================================== #
with open(os.path.join(stor_dir, 'params_stats.txt'), 'a') as f:
    f.write('{:.8f} {:8f} {:8f} {:8f} {:8f} {:8f}\n'.format(infilt, d1, d2, d3, expt, Ksat))
    f.write('{:8f} {:8f}\n'.format(kge_daily, kge_5D))


