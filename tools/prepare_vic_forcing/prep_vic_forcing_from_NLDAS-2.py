
import sys
import pandas as pd
import os
import xarray as xr
import numpy as np

from tonic.io import read_config, read_configobj
from prep_forcing_utils import to_netcdf_forcing_file_compress


# ======================================================= #
# Process command line argument
# ======================================================= #
cfg = read_configobj(sys.argv[1])


# ======================================================= #
# Process data for each VIC timestep
# ======================================================= #
print('Processing NLDAS-2 hourly data...')
# --- Loop over each VIC timestep (if > hourly, each VIC timestep --- #
# --- will contain multiple hourly NLDAS-2 data) --- #
vic_times = pd.date_range(cfg['TIME']['start_time'],
                      cfg['TIME']['end_time'],
                      freq='{}H'.format(cfg['VIC']['time_step']))
# Calculate CONST_EPS for vapor pressure calculate
CONST_EPS = 18.016 / 28.97  # This is consisten with VIC
# Initialize a dictionary to store VIC-timestep forcing data
# Each element in the dict will be a list for one year
start_year = vic_times[0].year
end_year = vic_times[-1].year
dict_force_yearly_vic_timestep = {}  # This dict will save forcing data
dict_times_yearly_vic_timestep = {}  # This dict will save time indices
for year in range(start_year, end_year+1):
    dict_force_yearly_vic_timestep[year] = []
    dict_times_yearly_vic_timestep[year] = []
    
# --- Loop over each VIC timestep --- #
for vic_time in vic_times:
    print('\t', vic_time)
    current_year = vic_time.year
    # --- Loop over each hour in this VIC timestep and load NLDAS hourly data --- #
    list_ds = []
    for hour in range(0, int(cfg['VIC']['time_step'])):
        time = vic_time + pd.DateOffset(seconds=3600*hour)
        filename = os.path.join(
            cfg['NLDAS']['subset_dir'], str(current_year),
            'force.{}.{:02d}.nc'.format(time.strftime('%Y%m%d'), time.hour))
        # Load data
        ds = xr.open_dataset(filename)
        list_ds.append(ds)
    ds_vic_time = xr.concat(list_ds, dim='time')
    # --- Aggregate each variable --- #
    # Precipitation - sum; [kg/m^2] -> [mm/step]
    da_prec = ds_vic_time['var61'].sum(dim='time')
    da_prec.attrs['unit'] = 'mm/step'
    # Air temperature - mean; [K] -> [C]
    da_airT = ds_vic_time['var11'].mean(dim='time')[0, :, :] - 273.15
    da_airT.attrs['unit'] = 'C'
    # Air pressure - mean; [Pa] -> [kPa]
    da_pressure = ds_vic_time['var1'].mean(dim='time') / 1000
    da_pressure.attrs['unit'] = 'kPa'
    # Incoming shortwave - mean [W/m2] -> [W/m2]
    da_shortwave = ds_vic_time['var204'].mean(dim='time')
    da_shortwave.attrs['unit'] = 'W/m2'
    # Incoming longwave - mean [W/m2] -> [W/m2]
    da_longwave = ds_vic_time['var205'].mean(dim='time')
    da_longwave.attrs['unit'] = 'W/m2'
    # Vapor pressure - mean [-] -> [kPa]
    # Use approximate conversion from specific humidity to vapor pressure:
    # vp = q * p / CONST_EPS
    # This is consistent with VIC
    da_vp = ds_vic_time['var51'][:, 0, :, :] * ds_vic_time['var1'] / CONST_EPS / 1000
    da_vp = da_vp.mean(dim='time')
    da_vp.attrs['unit'] = 'kPa'
    # Wind speed - mean
    da_wind = np.sqrt(np.square(ds_vic_time['var33'][:, 0, :, :]) + \
                      np.square(ds_vic_time['var34'][:, 0, :, :]))
    da_wind = da_wind.mean(dim='time')
    da_wind.attrs['unit'] = 'm/s'
    # Put all variables together to a dataset
    ds_force_vic_time = xr.Dataset(
        {'PREC': da_prec,
         'AIR_TEMP': da_airT,
         'PRESSURE': da_pressure,
         'SHORTWAVE': da_shortwave,
         'LONGWAVE': da_longwave,
         'VP': da_vp,
         'WIND': da_wind})
    # Store dataset to the list of corresponding year
    dict_force_yearly_vic_timestep[current_year].append(ds_force_vic_time)
    dict_times_yearly_vic_timestep[current_year].append(vic_time)


# ======================================================= #
# Concat each year of forcing
# ======================================================= #
print('Concatenating timesteps...')
ds_force_yearly = {}  # Dict keys are year
for year in range(start_year, end_year+1):
    list_force_vic_timestep = dict_force_yearly_vic_timestep[year]
    ds_concat_yearly = xr.concat(list_force_vic_timestep, dim='time')
    ds_concat_yearly['time'] = dict_times_yearly_vic_timestep[year]
    ds_force_yearly[year] = ds_concat_yearly


# ======================================================= #
# Remap to the VIC parameter grid
# Since NLDAS-2 grid is 0.0005 degree off with VIC 1/8th
# Parameter files, regrid NLDAS-2 data using nearest neighbors
# (In practive, here we simply shift the lat and lon of
# the NLDAS-2 data)
# ======================================================= #
print('Remapping to VIC grid...')
# --- Shift lat and lon in the NLDAS forcing data --- #
lats = ds_force_yearly[start_year]['lat'].values - 0.0005
lons = ds_force_yearly[start_year]['lon'].values + 0.0005

for year in range(start_year, end_year+1):
    times = ds_force_yearly[year]['time'].values
    ds_new = xr.Dataset(
        {'PREC': (['time', 'lat', 'lon'], ds_force_yearly[year]['PREC']),
         'AIR_TEMP': (['time', 'lat', 'lon'], ds_force_yearly[year]['AIR_TEMP']),
         'PRESSURE': (['time', 'lat', 'lon'], ds_force_yearly[year]['PRESSURE']),
         'SHORTWAVE': (['time', 'lat', 'lon'], ds_force_yearly[year]['SHORTWAVE']),
         'LONGWAVE': (['time', 'lat', 'lon'], ds_force_yearly[year]['LONGWAVE']),
         'VP': (['time', 'lat', 'lon'], ds_force_yearly[year]['VP']),
         'WIND': (['time', 'lat', 'lon'], ds_force_yearly[year]['WIND']),},
        coords={'time': (['time'], times),
                'lat': (['lat'], lats),
                'lon': (['lon'], lons)})
    ds_force_yearly[year] = ds_new


# ======================================================= #
# Mask forcing data with domain file
# ======================================================= #
print('Masking...')
# --- Load domain file --- #
ds_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])
mask_var = cfg['DOMAIN']['mask_name']
lat_name = cfg['DOMAIN']['lat_name']
lon_name = cfg['DOMAIN']['lon_name']
# --- Mask --- #
for year in range(start_year, end_year+1):
    ds_force_yearly[year] = ds_force_yearly[year].where(ds_domain[mask_var].values)


# ======================================================= #
# Save final forcing data to file
# ======================================================= #
print('Saving to file...')
for year in range(start_year, end_year+1):
    to_netcdf_forcing_file_compress(
        ds_force_yearly[year],
        os.path.join(cfg['OUTPUT']['out_dir'], 'force.{}.nc'.format(year)))


