
''' This script converts Newman ensemble forcing data to VIC5 (image) input format.
    Note that prec in Newman is directly used; Tmax and Tmin are calcualted from
    t_mean and t_range in Newman dataset (Tmax = t_mean + t_range / 2, Tmin =
    t_mean - t_range / 2); wind speed is taken from Maurer data.
    Specific steps:
        (1) Select out target domain from orig. Maurer netCDF forcing files, as
            well as each Newman ensemble member (if domain of which is not already
            extracted)
        (2) Convert Newman and Maurer forcings to ascii format for each ensemble
            member
        (3) Run VIC4.2 as met disaggregator to get 7 required met variables,
            subdaily, for each ensemble member
        (4) Convert results to netCDF format, for each ensemble member

   Usage:
        $ python prep_vic_forcing_from_Newman.py <config_file> <nproc>
        where <config_file> is the config file, <nproc> is the number of processors to use
'''

import xarray as xr
import sys
import pandas as pd
import os
from collections import OrderedDict
import subprocess
import string
import multiprocessing as mp

from tonic.io import read_config, read_configobj
from tonic.models.vic.vic2netcdf import vic2nc
from tonic.models.vic.vic import VIC
from tonic.models.vic.vic import VIC, default_vic_valgrind_error_code


# ------------------------------------------------------------------- #
def main(cfg_file, nproc=1):
    ''' Main function

    Parameters
    ----------
    cfg_file: <str>
        Input config file
    nproc: <int>
        Number of processors to use
    '''
    
    # ====================================================== #
    # Load in config file
    # ====================================================== #
    cfg = read_configobj(cfg_file)
   
 
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
                             mkdirs=['forc_orig_nc', 'forc_orig_asc',
                                     'forc_disagg_asc', 'forc_disagg_nc',
                                     'config_files', 'logs_vic'])
    # Subdirs for config files for ensemble
    subdirs_config = setup_output_dirs(
                            dirs['config_files'],
                            mkdirs=['netcdf2vic', 'vic4', 'vic2nc'])
    
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
    # Load in and process Newman ensemble forcings (for prec, Tmax and Tmin)
    # and orig. Maurer forcing (for wind speed)
    # ====================================================== #
    
    # --- Load Maurer forcings --- #
    print('Processing Maurer forcings...')
    
    # Loop over each year
    list_da_wind = []
    for year in range(start_year, end_year+1):
        print('Year {}'.format(year))
        # --- Load in netCDF file for this year --- #
        da_wind = xr.open_dataset(os.path.join(
                        cfg['FORCING']['maurer_dir'],
                        'nldas_met_update.obs.daily.wind.{}.nc'.format(year)))['wind']
        # --- Mask out the target area --- #
        da_wind = da_wind.sel(latitude=slice(lat_min, lat_max),
                              longitude=slice(lon_min, lon_max))
        da_wind = da_wind.where(da_domain.values)
        # --- Rename lat and lon --- #
        da_wind = da_wind.rename({'latitude': 'lat', 'longitude': 'lon'})
        # --- Put in list --- #
        list_da_wind.append(da_wind)
    
    # Concat all years together
    da_wind_allyears = xr.concat(list_da_wind, dim='time')
   
    # --- Load Newman forcings --- #
    print('Processing Newman forcings...')

    # If 1 processor, do a regular process
    if nproc == 1:
        # Loop over each ensemble member
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            load_and_process_Newman(ens, cfg, da_domain, lat_min, lat_max,
                                    lon_min, lon_max, start_date, end_date,
                                    dirs, da_wind_allyears)
    # If multiple processors, use mp
    elif nproc > 1:
        # Set up multiprocessing
        pool = mp.Pool(processes=nproc)
        # Loop over each ensemble member
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            pool.apply_async(load_and_process_Newman,
                             (ens, cfg, da_domain, lat_min, lat_max, lon_min,
                              lon_max, start_date, end_date, dirs,
                              da_wind_allyears,))
        # Finish multiprocessing
        pool.close()
        pool.join()
    
    # ====================================================== #
    # Convert orig. forcings to ascii format
    # ====================================================== #
    
    print('Converting orig. netCDF forcings to VIC ascii...')

    # --- Setup subdirs for asc VIC orig. forcings for each ensemble member
    # --- #
    list_ens = []
    for ens in range(1, cfg['FORCING']['n_ens']+1):
        list_ens.append('ens_{}'.format(ens))
    subdirs_output = setup_output_dirs(
                        dirs['forc_orig_asc'],
                        mkdirs=list_ens)
    
    # --- Prepare netcdf2vic config file --- #
    dict_cfg_file = {}
    for ens in range(1, cfg['FORCING']['n_ens']+1):
        cfg_file = os.path.join(subdirs_config['netcdf2vic'],
                                'ens_{}.cfg'.format(ens))
        dict_cfg_file[ens] = cfg_file

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
            f.write('latlon_precision: {}\n'.format(
                            cfg['OUTPUT']['latlon_precision']))

            f.write('\n[paths]\n')
            f.write('in_path: {}\n'.format(os.path.join(
                                        dirs['forc_orig_nc'],
                                        'ens_{}'.format(ens))))
            f.write('mask_path: {}\n'.format(cfg['DOMAIN']['domain_nc']))
            f.write('mask_varname: {}\n'.format(cfg['DOMAIN']['mask_name']))
            f.write('ASCIIoutPath: {}\n'.format(
                        subdirs_output['ens_{}'.format(ens)]))
        
    # --- Run nc_to_vic --- #
    # If 1 processor, do a regular process
    if nproc == 1:
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            nc_to_vic(dict_cfg_file[ens])
    # If multiple processors, use mp
    elif nproc > 1:
        # Set up multiprocessing
        pool = mp.Pool(processes=nproc)
        # Loop over each ensemble member
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            pool.apply_async(nc_to_vic, (dict_cfg_file[ens],))
        # Finish multiprocessing
        pool.close()
        pool.join()
    
    # ====================================================== #
    # Run VIC forcing disaggregator
    # ====================================================== #
    
    print('Running VIC as a disaggregator...')
    
    # --- Setup subdirs for asc VIC disagg. forcings and VIC log files for
    # each ensemble member --- #
    list_ens = []
    for ens in range(1, cfg['FORCING']['n_ens']+1):
        list_ens.append('ens_{}'.format(ens))
    subdirs_output = setup_output_dirs(
                        dirs['forc_disagg_asc'],
                        mkdirs=list_ens)
    subdirs_logs = setup_output_dirs(
                        dirs['logs_vic'],
                        mkdirs=list_ens)
 
    # --- Prepare VIC global file for the disaggregation run --- #
    # Load in global file template
    with open(cfg['VIC_DISAGG']['global_template'], 'r') as f:
         global_param = f.read()
    # Create string template
    s = string.Template(global_param)
    # Loop over each ensemble member
    dict_global_file = {}
    for ens in range(1, cfg['FORCING']['n_ens']+1):
        # Fill in variables in the template
        global_param = s.safe_substitute(
                            time_step=cfg['VIC_DISAGG']['time_step'],
                            startyear=start_year,
                            startmonth=start_date.month,
                            startday=start_date.day,
                            endyear=end_year,
                            endmonth=end_date.month,
                            endday=end_date.day,
                            forcing1=os.path.join(dirs['forc_orig_asc'],
                                                  'ens_{}'.format(ens),
                                                  'forc_orig_'),
                            grid_decimal=cfg['OUTPUT']['latlon_precision'],
                            prec='PREC',
                            tmax='TMAX',
                            tmin='TMIN',
                            wind='WIND',
                            forceyear=start_year,
                            forcemonth=start_date.month,
                            forceday=start_date.day,
                            result_dir=subdirs_output['ens_{}'.format(ens)])
        # Write global param file
        global_file = os.path.join(subdirs_config['vic4'],
                                   'vic.global.ens_{}.txt'.format(ens))
        dict_global_file[ens] = global_file
        with open(global_file, mode='w') as f:
            for line in global_param:
                f.write(line)
            
    # --- Run VIC --- #
    # Prepare VIC exe
    vic_exe = VIC(cfg['VIC_DISAGG']['vic4_exe'])

    # If 1 processor, do a regular process
    if nproc == 1:
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            vic_exe.run(dict_global_file[ens],
                        logdir=subdirs_logs['ens_{}'.format(ens)])
    # If multiple processors, use mp
    elif nproc > 1:
        # Set up multiprocessing
        pool = mp.Pool(processes=nproc)
        # Loop over each ensemble member
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            pool.apply_async(run_vic_for_multiprocess,
                             (vic_exe, dict_global_file[ens],
                              subdirs_logs['ens_{}'.format(ens)],))
        # Finish multiprocessing
        pool.close()
        pool.join()
    
    # ====================================================== #
    # Convert disaggregated forcings to netCDF format
    # ====================================================== #
    
    # --- Prepare config file for vic2nc --- #
    print('Converting disaggregated forcings to netCDF...')
    
    # --- Setup subdirs for VIC disagg. netCDF forcings for each ensemble
    # member --- #
    list_ens = []
    for ens in range(1, cfg['FORCING']['n_ens']+1):
        list_ens.append('ens_{}'.format(ens))
    subdirs_output = setup_output_dirs(
                        dirs['forc_disagg_nc'],
                        mkdirs=list_ens)

    # --- Prepare netcdf2vic config file --- #
    # Extract disaggregated forcing variable names and order
    with open(cfg['VIC_DISAGG']['global_template'], 'r') as f:
         global_param = f.read()
    outvar_list = find_outvar_global_param(global_param)
    for i, var in enumerate(outvar_list):
        outvar_list[i] = var.strip('OUT_')
   
    # Extract end date and hour 
    end_date_with_hour = end_date + pd.DateOffset(days=1) -\
                         pd.DateOffset(hours=cfg['VIC_DISAGG']['time_step'])

    # Loop over each ensemble member 
    dict_cfg_file = {}
    for ens in range(1, cfg['FORCING']['n_ens']+1):
        cfg_file = os.path.join(subdirs_config['vic2nc'],
                                'ens_{}.cfg'.format(ens))
        dict_cfg_file[ens] = cfg_file
        
        with open(cfg_file, 'w') as f:
            f.write('[OPTIONS]\n')
            f.write('input_files: {}\n'.format(
                        os.path.join(dirs['forc_disagg_asc'],
                                     'ens_{}'.format(ens),
                                     'force_*')))
            f.write('input_file_format: ascii\n')
            f.write('bin_dt_sec: {}\n'.format(cfg['VIC_DISAGG']['time_step']*3600))
            f.write('bin_start_date: {}\n'.format(start_date.strftime("%Y-%m-%d-%H")))
            f.write('bin_end_date: {}\n'.format(end_date_with_hour.strftime("%Y-%m-%d-%H")))
            f.write('regular_grid: False\n')
            f.write('out_directory: {}\n'.format(subdirs_output['ens_{}'.format(ens)]))
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
                elif var == 'LONGAVE':
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
    # If 1 processor, do a regular process
    if nproc == 1:
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            cfg_vic2nc = read_config(dict_cfg_file[ens])
            options = cfg_vic2nc.pop('OPTIONS')
            global_atts = cfg_vic2nc.pop('GLOBAL_ATTRIBUTES')
            if not options['regular_grid']:
                domain_dict = cfg_vic2nc.pop('DOMAIN')
            else:
                domain_dict = None
            # Set aside fields dict
            fields = cfg_vic2nc
            # Run vic2nc 
            vic2nc(options, global_atts, domain_dict, fields)

    # If multiple processors, use mp
    elif nproc > 1:
        # Set up multiprocessing
        pool = mp.Pool(processes=nproc)
        # Loop over each ensemble member
        for ens in range(1, cfg['FORCING']['n_ens']+1):
            cfg_vic2nc = read_config(dict_cfg_file[ens])
            options = cfg_vic2nc.pop('OPTIONS')
            global_atts = cfg_vic2nc.pop('GLOBAL_ATTRIBUTES')
            if not options['regular_grid']:
                domain_dict = cfg_vic2nc.pop('DOMAIN')
            else:
                domain_dict = None
            # set aside fields dict
            fields = cfg_vic2nc
            pool.apply_async(vic2nc,
                             (options, global_atts, domain_dict, fields,))
        # Finish multiprocessing
        pool.close()
        pool.join()
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
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
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
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
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
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
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
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
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
def load_and_process_Newman(ens, cfg, da_domain, lat_min, lat_max, lon_min,
                            lon_max, start_date, end_date, dirs,
                            da_wind_allyears):
    ''' Load and process Newman data (prec and temperature) for one ensemble
        member, combine with wind speed from Maurer and save to file.

    Parameters
    ----------
    ens: <int>
        Ensemble index (start from 1)
    cfg: <configobj.ConfigObj>
        Config file to the main function
    da_domain: <xr.DataArray>
        Domain mask
    lat_min, lat_max, lon_min, lon_max: <float>
        Range of domain file
    start_date, end_date: <pandas.tslib.Timestamp>
        Date range of process
    dirs: <dict>
        Uppder-level output directory dict
    da_wind_allyears: <xr.DataArray>
        Wind data from Maurer for all target years, domain extracted

    Require
    ----------
    xarray
    '''

    print('Load and process Newman data for ensemble {}'.format(ens))  
    
    # --- Load in netCDF file for this ensemble member --- #
    ds = xr.open_dataset('{}{:03d}.nc'.format(
                    cfg['FORCING']['newman_basepath'], ens))
    # --- Mask out target domain and period of time --- #
    ds = ds.sel(lat=slice(lat_min, lat_max),
                lon=slice(lon_min, lon_max),
                time=slice(start_date, end_date))
    ds = ds.where(da_domain.values)
    # --- Extract prec and calculate Tmax and Tmin --- #
    # Prec
    da_pr = ds['pcp']
    da_pr.attrs['units'] = 'mm/d'
    # Tmax and Tmin
    da_t_mean = ds['t_mean']
    da_t_range = ds['t_range']
    da_tasmax = da_t_mean + da_t_range / 2
    da_tasmin = da_t_mean - da_t_range / 2
    # --- Combine all variables --- #
    ds_combine = xr.Dataset({"pr": da_pr, "tasmax": da_tasmax, "tasmin": da_tasmin,
                             "wind": da_wind_allyears})
    # --- Make subdir for output netCDF files --- #
    subdir_ens = setup_output_dirs(
                        dirs['forc_orig_nc'],
                        mkdirs=['ens_{}'.format(ens)])['ens_{}'.format(ens)]
    # --- Write out a netCDF file, one file for each year --- #
    gb = ds_combine.groupby('time.year')
    for year, ds in gb:
        ds.to_netcdf(os.path.join(subdir_ens,
                                  'forc_orig.{}.nc'.format(year)),
                     format='NETCDF4_CLASSIC')
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
def check_returncode(returncode, expected=0):
    '''check return code given by VIC, raise error if appropriate

    Require
    ---------
    tonic.models.vic.vic.default_vic_valgrind_error_code
    class VICReturnCodeError
    '''
    if returncode == expected:
        return None
    elif returncode == default_vic_valgrind_error_code:
        raise VICValgrindError('Valgrind raised an error')
    else:
        raise VICReturnCodeError('VIC return code ({0}) does not match '
                                 'expected ({1})'.format(returncode, expected))
# ------------------------------------------------------------------- #


# ------------------------------------------------------------------- #
def run_vic_for_multiprocess(vic_exe, global_file, log_dir):
    '''This function is a simple wrapper for calling "run" method under
        VIC class in multiprocessing

    Parameters
    ----------
    vic_exe: <class VIC>
        A VIC class object
    global_file: <str>
        VIC global file path
    log_dir: <str>
        VIC run output log directory

    Require
    ----------
    check_returncode
    '''

    returncode = vic_exe.run(global_file, logdir=log_dir)
    check_returncode(returncode, expected=0)
# ------------------------------------------------------------------- #


if __name__ == "__main__":
    main(cfg_file=sys.argv[1], nproc=int(sys.argv[2]))
    
