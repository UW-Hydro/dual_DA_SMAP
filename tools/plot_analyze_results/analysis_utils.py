
import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import sys
import multiprocessing as mp
from collections import OrderedDict
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader

from tonic.io import read_configobj
import timeit


def rmse(true, est):
    ''' Calculates RMSE of an estimated variable compared to the truth variable

    Parameters
    ----------
    true: <np.array>
        A 1-D array of time series of true values
    est: <np.array>
        A 1-D array of time series of estimated values (must be the same length of true)

    Returns
    ----------
    rmse: <float>
        Root mean square error

    Require
    ----------
    numpy
    '''

    rmse = np.sqrt(sum((est - true)**2) / len(true))
    return rmse


def find_global_param_value(gp, param_name, second_param=False):
    ''' Return the value of a global parameter

    Parameters
    ----------
    gp: <str>
        Global parameter file, read in by read()
    param_name: <str>
        The name of the global parameter to find
    second_param: <bool>
        Whether to read a second value for the parameter (e.g., set second_param=True to
        get the snowband param file path when SNOW_BAND>1)

    Returns
    ----------
    line_list[1]: <str>
        The value of the global parameter
    (optional) line_list[2]: <str>
        The value of the second value in the global parameter file when second_param=True
    '''
    for line in iter(gp.splitlines()):
        line_list = line.split()
        if line_list == []:
            continue
        key = line_list[0]
        if key == param_name:
            if second_param == False:
                return line_list[1]
            else:
                return line_list[1], line_list[2]


def calculate_max_soil_moist_domain(global_path):
    ''' Calculates maximum soil moisture for all grid cells and all soil layers (from soil parameters)

    Parameters
    ----------
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract soil parameter file info)

    Returns
    ----------
    da_max_moist: <xarray.DataArray>
        Maximum soil moisture for the whole domain and each soil layer [unit: mm];
        Dimension: [nlayer, lat, lon]

    Require
    ----------
    xarray
    find_global_param_value
    '''
    # Load soil parameter file (as defined in global file)
    with open(global_path, 'r') as global_file:
        global_param = global_file.read()
    soil_nc = find_global_param_value(global_param, 'PARAMETERS')
    ds_soil = xr.open_dataset(soil_nc, decode_cf=False)

    # Calculate maximum soil moisture for each layer
    # Dimension: [nlayer, lat, lon]
    da_depth = ds_soil['depth']  # [m]
    da_bulk_density = ds_soil['bulk_density']  # [kg/m3]
    da_soil_density = ds_soil['soil_density']  # [kg/m3]
    da_porosity = 1 - da_bulk_density / da_soil_density
    da_max_moist = da_depth * da_porosity * 1000  # [mm]

    return da_max_moist


def determine_tile_frac(global_path):
    ''' Determines the fraction of each veg/snowband tile in each grid cell based on VIC
        global and parameter files

    Parameters
    ----------
    global_path: <str>
        VIC global parameter file path; can be a template file (here it is only used to
        extract snowband and vegparam files/options)

    Returns
    ----------
    da_tile_frac: <xr.DataArray>
        Fraction of each veg/snowband in each grid cell for the whole domain
        Dimension: [veg_class, snow_band, lat, lon]

    Require
    ----------
    numpy
    xarray
    '''

    # --- Load global parameter file --- #
    with open(global_path, 'r') as global_file:
            global_param = global_file.read()

    # --- Extract Cv from vegparam file (as defined in the global file) --- #
    param_nc = find_global_param_value(global_param, 'PARAMETERS')
    ds_param = xr.open_dataset(param_nc, decode_cf=False)
    da_Cv = ds_param['Cv']  # dim: [veg_class, lat, lon]
    lat = da_Cv['lat']
    lon = da_Cv['lon']

    # --- Extract snowband info from the global and param files --- #
    SNOW_BAND = find_global_param_value(global_param, 'SNOW_BAND')
    if SNOW_BAND.upper() == 'TRUE':
        n_snowband = len(ds_param['snow_band'])
    else:
        n_snowband = 1
    # Dimension of da_AreaFract: [snowband, lat, lon]
    if n_snowband == 1:  # if only one snowband
        data = np.ones([1, len(lat), len(lon)])
        da_AreaFract = xr.DataArray(data, coords=[[0], lat, lon],
                                    dims=['snow_band', 'lat', 'lon'])
    else:  # if more than one snowband
        da_AreaFract = ds_param['AreaFract']

    # --- Initialize the final DataArray --- #
    veg_class = da_Cv['veg_class']
    snow_band = da_AreaFract['snow_band']
    data = np.empty([len(veg_class), len(snow_band), len(lat), len(lon)])
    data[:] = np.nan
    da_tile_frac = xr.DataArray(data, coords=[veg_class, snow_band, lat, lon],
                                dims=['veg_class', 'snow_band', 'lat', 'lon'])

    # --- Calculate fraction of each veg/snowband tile for each grid cell,
    # and fill in da_file_frac --- #
    # Determine the total number of loops
    nloop = len(lat) * len(lon)
    # Convert Cv and AreaFract to np.array and straighten lat and lon into nloop
    Cv = da_Cv.values.reshape([len(veg_class), nloop])  # [nveg, nloop]
    AreaFract = da_AreaFract.values.reshape([len(snow_band), nloop])  # [nsnow, nloop]

    # Multiply Cv and AreaFract for each tile and grid cell
    tile_frac = np.array(list(map(
                    lambda i: np.dot(
                        Cv[:, i].reshape([len(veg_class), 1]),
                        AreaFract[:, i].reshape([1, len(snow_band)])),
                    range(nloop))))  # [nloop, nveg, nsnow]

    # Reshape tile_frac
    tile_frac = np.rollaxis(tile_frac, 0, 3)  # [nveg, nsow, nloop]
    tile_frac = tile_frac.reshape([len(veg_class), len(snow_band), len(lat), len(lon)])

    # Put in da_tile_frac
    da_tile_frac[:] = tile_frac

    return da_tile_frac


def get_soil_depth(param_nc):
    '''Get soil depth for all grid cells and all soil layers (from soil parameters)

    Parameters
    ----------
    param_nc: <str>
        VIC input parameter netCDF file path (here it is only used to
        extract soil parameter file info)

    Returns
    ----------
    da_soil_depth: <xarray.DataArray>
        Soil depth for the whole domain and each soil layer [unit: mm];
        Dimension: [nlayer, lat, lon]

    Require
    ----------
    xarray
    '''

    # Load soil parameter file
    ds_soil = xr.open_dataset(param_nc)

    # Get soil depth for each layer
    # Dimension: [nlayer, lat, lon]
    da_soil_depth = ds_soil['depth']  # [m]
    # Convert unit to mm
    da_soil_depth = da_soil_depth * 1000 # [mm]

    return da_soil_depth


def load_nc_file(nc_file, start_year, end_year):
    ''' Loads in nc files for all years.

    Parameters
    ----------
    nc_file: <str>
        netCDF file to load, with {} to be substituted by YYYY
    start_year: <int>
        Start year
    end_year: <int>
        End year

    Returns
    ----------
    ds_all_years: <xr.Dataset>
        Dataset of all years
    '''

    list_ds = []
    for year in range(start_year, end_year+1):
        # Load data
        fname = nc_file.format(year)
        ds = xr.open_dataset(fname)
        list_ds.append(ds)
        # Concat all years
        ds_all_years = xr.concat(list_ds, dim='time')

    return ds_all_years


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


def calc_sm_runoff_corrcoef(sm, runoff):
    runoff_with_runoff = runoff[runoff>0]
    sm_with_runoff = sm[runoff>0]

    return np.corrcoef(sm_with_runoff, runoff_with_runoff)[0, 1]


def to_netcdf_state_file_compress(ds_state, out_nc):
    ''' This function saves a VIC-state-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_state: <xr.Dataset>
        State dataset to save
    out_nc: <str>
        Path of output netCDF file
    '''

    dict_encode = {}
    for var in ds_state.data_vars:
        if var.split('_')[0] != 'STATE':
            continue
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1}
    ds_state.to_netcdf(out_nc,
                       format='NETCDF4',
                       encoding=dict_encode)


def calculate_rmse(out_nc, ds_truth, ds_model,
                   var, depth_sm=None):
    ''' A wrap funciton that calculates RMSE for all domain and save to file; if
        result file already existed, then simply read in the file.
    
    Parameters
    ----------
    out_nc: <str>
        RMSE result output netCDF file
    ds_truth: <xr.Dataset>
        Truth states/history
    ds_model: <xr.Dataset>
        Model states/history whose RMSE is to be assessed (wrt. truth states)
    var: <str>
        Variable, options:
            sm1; sm2; sm3; runoff_daily; baseflow_daily; totrunoff_daily; swe;
            runoff_daily_log; baseflow_daily_log; totrunoff_daily_log
        NOTE: sm's and swe are from states; runoff's are from history file
    depth_sm: <xr.DataArray>
        Thickness of soil moisture
        Only required if state_var is soil moisture
    
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
        elif var == 'swe':
            da_truth = ds_truth['SWE']
            da_model = ds_model['SWE']
        elif var == 'runoff_daily':
            da_truth = ds_truth['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum')
            da_model = ds_model['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum')
        elif var == 'baseflow_daily':
            da_truth = ds_truth['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum')
            da_model = ds_model['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum')
        elif var == 'totrunoff_daily':
            da_truth = ds_truth['OUT_RUNOFF'].resample('1D', dim='time', how='sum') + \
                       ds_truth['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
            da_model = ds_model['OUT_RUNOFF'].resample('1D', dim='time', how='sum') + \
                       ds_model['OUT_BASEFLOW'].resample('1D', dim='time', how='sum')
        elif var == 'runoff_daily_log':
            da_truth = np.log(ds_truth['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum') + 1)
            da_model = np.log(ds_model['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum') + 1)
        elif var == 'baseflow_daily_log':
            da_truth = np.log(ds_truth['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum') + 1)
            da_model = np.log(ds_model['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum') + 1)
        elif var == 'totrunoff_daily_log':
            da_truth = np.log(
                ds_truth['OUT_RUNOFF'].resample('1D', dim='time', how='sum') + \
                ds_truth['OUT_BASEFLOW'].resample('1D', dim='time', how='sum') +1)
            da_model = np.log(
                ds_model['OUT_RUNOFF'].resample('1D', dim='time', how='sum') + \
                ds_model['OUT_BASEFLOW'].resample('1D', dim='time', how='sum') + 1)
        # --- Calculate RMSE --- #
        # Determine the total number of loops
        lat_coord = da_truth['lat']
        lon_coord = da_truth['lon']
        nloop = len(lat_coord) * len(lon_coord)
        # Reshape variables
        truth = da_truth.values.reshape(
            [len(da_model['time']), nloop])  # [time, nloop]
        model = da_model.values.reshape(
            [len(da_model['time']), nloop])  # [time, nloop]
        # Calculate RMSE for all grid cells
        rmse_model = np.array(list(map(
                     lambda j: rmse(truth[:, j], model[:, j]),
                    range(nloop))))  # [nloop]
        # Reshape RMSE's
        rmse_model = rmse_model.reshape(
            [len(lat_coord), len(lon_coord)])  # [lat, lon]
        # Put results into da's
        da_rmse_model = xr.DataArray(
            rmse_model, coords=[lat_coord, lon_coord],
            dims=['lat', 'lon'])  # [mm/mm]
        # Save RMSE to netCDF file
        ds_rmse_model = xr.Dataset(
            {'rmse': da_rmse_model})
        ds_rmse_model.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
    else:  # if RMSE is already calculated
        da_rmse_model = xr.open_dataset(out_nc)['rmse']
    
    return da_rmse_model


def calculate_pbias(out_nc, ds_truth, ds_model,
                    var, depth_sm=None):
    ''' A wrap funciton that calculates PBIAS for all domain and save to file; if
        result file already existed, then simply read in the file.
    
    Parameters
    ----------
    out_nc: <str>
        RMSE result output netCDF file
    ds_truth: <xr.Dataset>
        Truth states/history
    ds_model: <xr.Dataset>
        Model states/history whose RMSE is to be assessed (wrt. truth states)
    var: <str>
        Variable, options:
            sm1; sm2; sm3; runoff_daily; baseflow_daily; totrunoff_daily; swe
        NOTE: sm's and sweare from states; runoff's are from history file
    depth_sm: <xr.DataArray>
        Thickness of soil moisture
        Only required if state_var is soil moisture
    
    '''
    
    if not os.path.isfile(out_nc):  # if PBIAS is not already calculated
        # --- Extract variables --- #
        if var == 'sm1':
            da_truth = ds_truth['SOIL_MOISTURE'].sel(nlayer=0) / depth_sm
            da_model = ds_model['SOIL_MOISTURE'].sel(nlayer=0) / depth_sm
        if var == 'sm2':
            da_truth = ds_truth['SOIL_MOISTURE'].sel(nlayer=1) / depth_sm
            da_model = ds_model['SOIL_MOISTURE'].sel(nlayer=1) / depth_sm
        if var == 'sm3':
            da_truth = ds_truth['SOIL_MOISTURE'].sel(nlayer=2) / depth_sm
            da_model = ds_model['SOIL_MOISTURE'].sel(nlayer=2) / depth_sm
        if var == 'swe':
            da_truth = ds_truth['SWE']
            da_model = ds_model['SWE']
        if var == 'runoff_daily':
            da_truth = ds_truth['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum')
            da_model = ds_model['OUT_RUNOFF'].resample(
                '1D', dim='time', how='sum')
        if var == 'baseflow_daily':
            da_truth = ds_truth['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum')
            da_model = ds_model['OUT_BASEFLOW'].resample(
                '1D', dim='time', how='sum')
        if var == 'totrunoff_daily':
            da_truth = (ds_truth['OUT_RUNOFF'] + ds_truth['OUT_BASEFLOW'])\
                .resample('1D', dim='time', how='sum')
            da_model = (ds_model['OUT_RUNOFF'] + ds_model['OUT_BASEFLOW'])\
                .resample('1D', dim='time', how='sum')
        # --- Calculate PBIAS --- #
        # Determine the total number of loops
        lat_coord = da_truth['lat']
        lon_coord = da_truth['lon']
        nloop = len(lat_coord) * len(lon_coord)
        # Reshape variables
        truth = da_truth.values.reshape(
            [len(da_model['time']), nloop])  # [time, nloop]
        model = da_model.values.reshape(
            [len(da_model['time']), nloop])  # [time, nloop]
        # --- Calculate PBIAS --- #
        da_truth_mean = da_truth.mean(dim='time')
        da_model_mean = da_model.mean(dim='time')
        da_pbias_model = (da_model_mean - da_truth_mean) / da_truth_mean * 100
        # Save PBIAS to netCDF file
        ds_pbias_model = xr.Dataset(
            {'pbias': da_pbias_model})
        ds_pbias_model.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
    else:  # if PBIAS is already calculated
        da_pbias_model = xr.open_dataset(out_nc)['pbias']
    
    return da_pbias_model


def add_gridlines(axis, xlocs=[-80, -90, -100, -110, -120],
                  ylocs=[30, 35, 40], alpha=1):
    gl = axis.gridlines(draw_labels=True, xlocs=xlocs, ylocs=ylocs,
                        alpha=alpha)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return gl

