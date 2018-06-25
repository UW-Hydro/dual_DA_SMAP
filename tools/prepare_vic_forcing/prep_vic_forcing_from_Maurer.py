
''' This script converts Maurer forcing data to VIC5 (image) input format.
    Specific steps:
        (1) Select out target domain from orig. Maurer netCDF forcing files
        (2) Convert Maurer forcings to ascii format
        (3) Run VIC4.2 as met disaggregator to get 7 required met variables,
            subdaily
        (4) Convert results to netCDF format

   Usage:
        $ python prep_vic_forcing_from_Maurer.py <config_file> 
'''

import xarray as xr
import sys
import pandas as pd
import os
from collections import OrderedDict
import subprocess
import string

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic2netcdf import vic2nc


def setup_output_dirs(out_basedir, mkdirs=['results', 'state',
                                            'logs', 'plots']):
    ''' This function creates output directories.

    Parameters
    ----------
    out_basedir: <str>
        Output base directory for all output files
    mkdirs: <list>
        A list of subdirectories to make

    Require
    ----------
    os
    OrderedDict

    Returns
    ----------
    dirs: <OrderedDict>
        A dictionary of subdirectories

    '''

    dirs = OrderedDict()
    for d in mkdirs:
        dirs[d] = os.path.join(out_basedir, d)

    for dirname in dirs.values():
        os.makedirs(dirname, exist_ok=True)

    return dirs


def write_ascii(array, point, out_prefix, path, precision, append, verbose=False):
    """
    Write an array to standard VIC ASCII output.
    """
    
    import numpy as np
    
    fname = out_prefix + '{:.{}f}'.format(point[0], precision) +\
            '_' + '{:.{}f}'.format(point[1], precision)
    out_file = os.path.join(path, fname)
    if append:
        f = open(out_file, 'ab')
    else:
        f = open(out_file, 'wb')

    if verbose:
        print('Writing ASCII Data to'.format(out_file))
    
    np.savetxt(f, array, fmt='%12.7g')
    f.close()
    return


def nc_to_vic(config_file):
    ''' This function converts netCDF files to VIC ascii format files.
        (This function is adapted from tonic)
        
        Parameters
        ----------
        config_file: <str>
            Path of config file for nc_to_vic
        
        Returns
        ----------
        
        Requires
        ----------
        write_binary
    '''
    
    import numpy as np
    import struct
    import os
    from tonic.io import read_netcdf, read_config
    from tonic.pycompat import pyzip
    
    config = read_config(config_file)
    files = config['options']['files']  # should contain "{}", which will be replaced by YYYY
    var_keys = config['options']['var_keys']
    output_format = config['options']['output_format']  # Binary or ASCII
    out_prefix = config['options']['out_prefix']
    verbose = config['options']['verbose']
    coord_keys = config['options']['coord_keys']  # varname of lon and lat in netCDF files
    lon_name = coord_keys[0]
    lat_name = coord_keys[1]
    start_year = config['options']['start_year']
    end_year = config['options']['end_year']
    latlon_precision = config['options']['latlon_precision']
    
    paths = config['paths']
    mask_varname = paths['mask_varname']

    mask = read_netcdf(paths['mask_path'], variables=['mask'])[0][mask_varname]
    yi, xi = np.nonzero(mask)
    print('found {0} points in mask file.'.format(len(yi)))

    xlist = []
    ylist = []
    pointlist = []
    append = False

    for i, year in enumerate(range(start_year, end_year+1)):
        print('Year {}'.format(year))
        fname = files.format(year)
        d = read_netcdf(os.path.join(paths['in_path'], fname),
                        verbose=verbose)[0]

        if i == 0:

            # find point locations
            xs = d[lon_name]
            ys = d[lat_name]
            posinds = np.nonzero(xs > 180)
            xs[posinds] -= 360
            print('adjusted xs lon minimum')

            for y, x in pyzip(yi, xi):
                active_flag = False
                for key in var_keys:
                    if (d[key][:, y, x].all() is np.ma.masked) \
                            or (mask[y, x] == 0):
                        active_flag = True
                if not active_flag:
                    point = (ys[y], xs[x])
                    xlist.append(x)
                    ylist.append(y)
                    pointlist.append(point)

        else:
            append = True

        for y, x, point in pyzip(ylist, xlist, pointlist):

            data = np.empty((d[var_keys[0]].shape[0], len(var_keys)))

            for j, key in enumerate(var_keys):
                data[:, j] = d[key][:, y, x]

            if output_format == 'Binary':
                write_binary(data * binary_mult, point, binary_type,
                             out_prefix, paths['BinaryoutPath'], append)
            if output_format == 'ASCII':
                write_ascii(data, point, out_prefix, paths['ASCIIoutPath'],
                            latlon_precision, append)
    return


def find_outvar_global_param(gp):
    ''' Return a list OUTVAR in order from a global parameter

    Parameters
    ----------
    gp: <str>
        Global parameter file, read in by read()

    Returns
    ----------
    ourvar_list: <list>
        A list of outvars in order
    '''
    
    outvar_list = []
    
    for line in iter(gp.splitlines()):
        line_list = line.split()
        if line_list == []:
            continue
        key = line_list[0]
        if key == 'OUTVAR':
            outvar_list.append(line_list[1])
    return outvar_list


# ====================================================== #
# Load in config file
# ====================================================== #
cfg = read_configobj(sys.argv[1])


# ====================================================== #
# Process some cfg variables
# ====================================================== #
start_date = pd.to_datetime(cfg['FORCING']['start_date'])
end_date = pd.to_datetime(cfg['FORCING']['end_date'])

start_year = start_date.year
end_year = end_date.year


# ====================================================== #
# Set up output directories
# ====================================================== #
dirs = setup_output_dirs(cfg['OUTPUT']['out_basedir'],
                         mkdirs=['forc_orig_nc', 'forc_orig_asc', 'forc_disagg_asc',
                                 'forc_disagg_nc', 'config_files'])


# ====================================================== #
# Load in domain file
# ====================================================== #
ds_domain = xr.open_dataset(cfg['DOMAIN']['domain_nc'])
da_domain = ds_domain[cfg['DOMAIN']['mask_name']]

lat_min = da_domain['lat'].min().values
lat_max = da_domain['lat'].max().values
lon_min = da_domain['lon'].min().values
lon_max = da_domain['lon'].max().values


# ====================================================== #
# Load in and process orig. Maurer forcing files
# ====================================================== #

print('Processing orig. Maurer forcings...')

# Loop over each year
for year in range(start_year, end_year+1):
    print('Year {}'.format(year))
    # --- Load in netCDF file for this year --- #
    da_pr = xr.open_dataset(
        cfg['FORCING']['maurer_orig_nc'].format(var='pr', YYYY=year))['pr']
    da_tasmax = xr.open_dataset(
        cfg['FORCING']['maurer_orig_nc'].format(var='tasmax', YYYY=year))['tasmax']
    da_tasmin = xr.open_dataset(
        cfg['FORCING']['maurer_orig_nc'].format(var='tasmin', YYYY=year))['tasmin']
    da_wind = xr.open_dataset(
        cfg['FORCING']['maurer_orig_nc'].format(var='wind', YYYY=year))['wind']
    # --- Clean up time coordinate to avoid tiny time difference --- #
    for da in [da_pr, da_tasmax, da_tasmin, da_wind]:
        da['time'] = da_pr['time']
    # --- Combine variables --- #
    ds = xr.Dataset({"pr": da_pr, "tasmax": da_tasmax, "tasmin": da_tasmin,
                     "wind": da_wind})
    # --- Mask out the target area --- #
    ds_small = ds.sel(lat=slice(lat_min, lat_max),
                      lon=slice(lon_min, lon_max))
    ds_masked = ds_small.where(da_domain.values)
    # --- Write out a single nc file --- #
    ds_masked.to_netcdf(os.path.join(dirs['forc_orig_nc'],
                                     'forc_orig.{}.nc'.format(year)),
                        format='NETCDF4_CLASSIC')


# ====================================================== #
# Convert orig. forcings to ascii format
# ====================================================== #

print('Converting orig. netCDF forcings to VIC ascii...')

# --- Prepare netcdf2vic config file --- #
cfg_file = os.path.join(dirs['config_files'], 'netcdf2vic.cfg')

with open(cfg_file, 'w') as f:
    f.write('[options]\n')
    f.write('files: forc_orig.{}.nc\n')
    f.write('verbose: True\n')
    f.write('output_format: ASCII\n')
    f.write('out_prefix: forc_orig_\n')
    f.write('coord_keys: lon,lat\n')
    f.write('var_keys: pr,tasmax,tasmin,wind\n')
    f.write('start_year: {}\n'.format(start_year))
    f.write('end_year: {}\n'.format(end_year))
    f.write('latlon_precision: {}\n'.format(cfg['OUTPUT']['latlon_precision']))
    
    f.write('\n[paths]\n')
    f.write('in_path: {}\n'.format(dirs['forc_orig_nc']))
    f.write('mask_path: {}\n'.format(cfg['DOMAIN']['domain_nc']))
    f.write('mask_varname: {}\n'.format(cfg['DOMAIN']['mask_name']))
    f.write('ASCIIoutPath: {}\n'.format(dirs['forc_orig_asc']))
    
# --- Run nc_to_vic --- #
nc_to_vic(cfg_file)


# ====================================================== #
# Run VIC forcing disaggregator
# ====================================================== #

print('Running VIC as a disaggregator...')

# --- Prepare VIC global file for the disaggregation run --- #
# Load in global file template
with open(cfg['VIC_DISAGG']['global_template'], 'r') as f:
     global_param = f.read()
# Create string template
s = string.Template(global_param)
# Fill in variables in the template
global_param = s.safe_substitute(time_step=cfg['VIC_DISAGG']['time_step'],
                                 startyear=start_year,
                                 startmonth=start_date.month,
                                 startday=start_date.day,
                                 endyear=end_year,
                                 endmonth=end_date.month,
                                 endday=end_date.day,
                                 forcing1=os.path.join(dirs['forc_orig_asc'],
                                                       'forc_orig_'),
                                 grid_decimal=cfg['OUTPUT']['latlon_precision'],
                                 prec='PREC',
                                 tmax='TMAX',
                                 tmin='TMIN',
                                 wind='WIND',
                                 forceyear=start_year,
                                 forcemonth=start_date.month,
                                 forceday=start_date.day,
                                 result_dir=dirs['forc_disagg_asc'])
# Write global param file
global_file = os.path.join(dirs['config_files'], 'vic.global.disagg.txt')
with open(global_file, mode='w') as f:
    for line in global_param:
        f.write(line)
        
# --- Run VIC --- #
subprocess.call('{} -g {}'.format(cfg['VIC_DISAGG']['vic4_exe'], global_file),
                shell=True)


# ====================================================== #
# Convert disaggregated forcings to netCDF format
# ====================================================== #

# --- Prepare config file for vic2nc --- #
print('Converting disaggregated forcings to netCDF...')

# --- Prepare netcdf2vic config file --- #
cfg_file = os.path.join(dirs['config_files'], 'vic2nc.cfg')

# Extract disaggregated forcing variable names and order
outvar_list = find_outvar_global_param(global_param)
for i, var in enumerate(outvar_list):
    outvar_list[i] = var.strip('OUT_')

end_date_with_hour = end_date + pd.DateOffset(days=1) -\
                     pd.DateOffset(hours=cfg['VIC_DISAGG']['time_step'])
    
with open(cfg_file, 'w') as f:
    f.write('[OPTIONS]\n')
    f.write('input_files: {}\n'.format(
                os.path.join(dirs['forc_disagg_asc'], 'force_*')))
    f.write('input_file_format: ascii\n')
    f.write('bin_dt_sec: {}\n'.format(cfg['VIC_DISAGG']['time_step']*3600))
    f.write('bin_start_date: {}\n'.format(start_date.strftime("%Y-%m-%d-%H")))
    f.write('bin_end_date: {}\n'.format(end_date_with_hour.strftime("%Y-%m-%d-%H")))
    f.write('regular_grid: False\n')
    f.write('out_directory: {}\n'.format(dirs['forc_disagg_nc']))
    f.write('memory_mode: big_memory\n')
    f.write('chunksize: 100\n')
    f.write('out_file_prefix: force\n')
    f.write('out_file_format: NETCDF4\n')
    f.write('precision: single\n')
    f.write('start_date: {}\n'.format(start_date.strftime("%Y-%m-%d-%H")))
    f.write('end_date: {}\n'.format(end_date_with_hour.strftime("%Y-%m-%d-%H")))
    f.write('calendar: proleptic_gregorian\n')
    f.write('time_segment: year\n')
    f.write('snow_bands: False\n')
    f.write('veg_tiles: False\n')
    f.write('soil_layers: False\n')
    
    f.write('\n[DOMAIN]\n')
    f.write('filename: {}\n'.format(cfg['DOMAIN']['domain_nc']))
    f.write('longitude_var: {}\n'.format(cfg['DOMAIN']['lon_name']))
    f.write('latitude_var: {}\n'.format(cfg['DOMAIN']['lat_name']))
    f.write('y_x_dims: {}, {}\n'.format(cfg['DOMAIN']['lat_name'],
                                        cfg['DOMAIN']['lon_name']))
    f.write('copy_vars: {}, {}, {}\n'.format(cfg['DOMAIN']['mask_name'],
                                             cfg['DOMAIN']['lat_name'],
                                             cfg['DOMAIN']['lon_name']))
    
    f.write('\n[GLOBAL_ATTRIBUTES]\n')
    f.write('title: VIC forcings\n')
    f.write('version: VIC4.2\n')
    f.write('grid: 1/8\n')
    
    for i, var in enumerate(outvar_list):
        if var == 'AIR_TEMP':
            f.write('\n[AIR_TEMP]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: C\n')
            f.write('standard_name: air_temperature\n')
            f.write('description: air temperature\n')
        elif var == 'PREC':
            f.write('\n[PREC]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: mm/step\n')
            f.write('standard_name: precipitation\n')
            f.write('description: precipitation\n')
        elif var == 'PRESSURE':
            f.write('\n[PRESSURE]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: kPa\n')
            f.write('standard_name: surface_air_pressure\n')
            f.write('description: near-surface atmospheric pressure\n')
        elif var == 'SHORTWAVE':
            f.write('\n[SHORTWAVE]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: W m-2\n')
            f.write('standard_name: incoming_shortwave_radiation\n')
            f.write('description: incoming shortwave radiation\n')
        elif var == 'LONGWAVE':
            f.write('\n[LONGWAVE]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: W m-2\n')
            f.write('standard_name: incoming_longwave_radiation\n')
            f.write('description: incoming longwave radiation\n')
        elif var == 'VP':
            f.write('\n[VP]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: kPa\n')
            f.write('standard_name: water_vapor_pressure\n')
            f.write('description: near surface vapor pressure\n')
        elif var == 'WIND':
            f.write('\n[WIND]\n')
            f.write('column: {}\n'.format(i))
            f.write('units: m/s\n')
            f.write('standard_name: surface_air_pressure\n')
            f.write('description: near-surface wind speed\n')

# --- Run vic2nc --- #
cfg_vic2nc = read_config(cfg_file)
options = cfg_vic2nc.pop('OPTIONS')
global_atts = cfg_vic2nc.pop('GLOBAL_ATTRIBUTES')
if not options['regular_grid']:
    domain_dict = cfg_vic2nc.pop('DOMAIN')
else:
    domain_dict = None

# set aside fields dict
fields = cfg_vic2nc

vic2nc(options, global_atts, domain_dict, fields)


