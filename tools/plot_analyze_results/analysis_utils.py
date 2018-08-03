
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
import xesmf as xe
from scipy.sparse import coo_matrix
import xesmf as xe
import properscoring as ps

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


def to_netcdf_history_file_compress(ds_hist, out_nc):
    ''' This function saves a VIC-history-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_hist: <xr.Dataset>
        History dataset to save
    out_nc: <str>
        Path of output netCDF file
    '''

    dict_encode = {}
    for var in ds_hist.data_vars:
        # skip variables not starting with "OUT_"
        if var.split('_')[0] != 'OUT':
            continue
        # determine chunksizes
        chunksizes = []
        for i, dim in enumerate(ds_hist[var].dims):
            if dim == 'time':  # for time dimension, chunksize = 1
                chunksizes.append(1)
            else:
                chunksizes.append(len(ds_hist[dim]))
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1,
                            'chunksizes': chunksizes}
    ds_hist.to_netcdf(out_nc,
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


def edges_from_centers(centers):
    ''' Return an array of grid edge values from grid center values
    Parameters
    ----------
    centers: <np.array>
        A 1-D array of grid centers. Typically grid-center lats or lons. Dim: [n]

    Returns
    ----------
    edges: <np.array>
        A 1-D array of grid edge values. Dim: [n+1]
    '''

    edges = np.zeros(len(centers)+1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])

    return edges


def map_ind_2D_to_1D(ind_2D_lat, ind_2D_lon, len_x):
    ''' Maps an index of a 2D array to an index in a flattened 1D array.
        The 2D domain [lat, lon] is flattened into 1D as 2D.reshape([lat*lon]).

    Parameters
    ----------
    ind_2D_lat: <int>
        y or lat index in the 2D array. Index starts from 0.
    ind_2D_lon: <int>
        y or lon index in the 2D array. Index starts from 0.
    len_x: <int>
        Length of x (lon) in the 2D domain. Index starts from 0.

    Returns
    ----------
    ind_1D: <int>
        Index in the flattened array. Index starts from 0.
    '''

    if ind_2D_lon < len_x:
        ind_1D = ind_2D_lat * len_x + ind_2D_lon
    else:
        raise ValueError('x or lon index in a 2D domain exceeds the dimension!')

    return(ind_1D)


def map_ind_1D_to_2D(ind_1D, len_x):
    ''' Maps an index of a flattened array to an index in a 2D domain.
        The 2D domain [lat, lon] is flattened into 1D as 2D.reshape([lat*lon]).

    Parameters
    ----------
    ind_1D: <int>
        Index in a flattened array. Index starts from 0.
    len_x: <int>
        Length of x (lon) in the 2D domain. Index starts from 0.

    Returns
    ----------
    ind_2D: (<int>, <int>)
        Index in the 2D array. Index starts from 0. (lat, lon) order.
    '''

    ind_2D_lat = ind_1D // len_x
    ind_2D_lon = ind_1D % len_x

    return(ind_2D_lat, ind_2D_lon)


def remap_con(reuse_weight, da_source, final_weight_nc, da_target_domain,
              da_source_domain=None,
              tmp_weight_nc=None, process_method=None):
    ''' Conservative remapping

    Parameters
    ----------
    reuse_weight: <bool>
        Whether to use an existing weight file directly, or to calculate weights
    da_source: <xr.DataArray>
        Source data. The dimension names must be "lat" and "lon".
    final_weight_nc: <str>
        If reuse_weight = False, path for outputing the final weight file;
        if reuse_weight = True, path for the weight file to use for regridding
    da_target_domain: <xr.DataArray>
        Domain of the target grid.
    da_source_domain: <xr.DataArray> (Only needed when reuse_weight = False)
        Domain of the source grid.
    tmp_weight_nc: <str> (Only needed when reuse_weight = False)
        Path for outputing the temporary weight file from xESMF
    process_method: (Only needed when reuse_weight = False)
        This option is not implemented yet (right now, there is only one way of processing
        the weight file). Can be extended to have the option of, e.g., setting a threshold
        for whether to remap for a target cell or not if the coverage is low

    Requires
    ----------
    process_weight_file
    import xesmf as xe
    '''

    # --- Grab coordinate information --- #
    src_lons = da_source['lon'].values
    src_lats = da_source['lat'].values
    target_lons = da_target_domain['lon'].values
    target_lats = da_target_domain['lat'].values
    lon_in_edges = edges_from_centers(src_lons)
    lat_in_edges = edges_from_centers(src_lats)
    lon_out_edges = edges_from_centers(target_lons)
    lat_out_edges = edges_from_centers(target_lats)
    grid_in = {'lon': src_lons,
               'lat': src_lats,
               'lon_b': lon_in_edges,
               'lat_b': lat_in_edges}
    grid_out = {'lon': target_lons,
                'lat': target_lats,
                'lon_b': lon_out_edges,
                'lat_b': lat_out_edges}
    # --- If reuse_weight = False, calculate weights --- #
    if reuse_weight is False:
        # Create regridder using xESMF
        regridder = xe.Regridder(grid_in, grid_out, 'conservative',
                                 filename=tmp_weight_nc)
        # Process the weight file to be correct, and save to final_weight_nc
        weight_array = process_weight_file(
            tmp_weight_nc, final_weight_nc,
            len(src_lons) * len(src_lats),
            len(target_lons) * len(target_lats),
            da_source_domain,
            process_method=None)  # weight_array: [n_target, n_source]
    else:
        print('Reusing weights: {}'.format(final_weight_nc))
    # --- Use the final weight file to regrid input data --- #
    # Load final weights
    n_source = len(src_lons) * len(src_lats)
    n_target = len(target_lons) * len(target_lats)
    A = xe.frontend.read_weights(final_weight_nc, n_source, n_target)
    weight_array = A.toarray()  # [n_target, n_source]
    # Apply weights to remap
    array_remapped = xe.frontend.apply_weights(
        A, da_source.values, len(target_lats), len(target_lons))
    # Track metadata
    varname = da_source.name
    extra_dims = da_source.dims[0:-2]
    extra_coords = [da_source.coords[dim].values for dim in extra_dims]
    da_remapped = xr.DataArray(
        array_remapped,
        dims=extra_dims + ('lat', 'lon'),
        coords=extra_coords + [target_lats, target_lons],
        name=varname)
    # If weight for a target cell is negative, it means that the target cell
    # does not overlap with any valid source cell. Thus set the remapped value to NAN
    nan_weights = (weight_array.sum(axis=1).reshape([len(target_lats), len(target_lons)]) < 0)
    data = da_remapped.values
    data[..., nan_weights] = np.nan
    da_remapped[:] = data

    return da_remapped, weight_array


def process_weight_file(orig_weight_nc, output_weight_nc, n_source, n_target,
                        da_source_domain, process_method=None):
    ''' Process the weight file generated by xESMF.
    Currently, use conservative remapping.
    Process rules:
        1) If a target grid cell is partially covered by source grid cells
        (regardless of the coverage), then those source cells will be scaled and
        used to remap to that target cell.
        2)

    Parameters
    ----------
    orig_weight_nc: <str>
        Original weight netCDF file output from xESMF
    output_weight_nc: <str>
        Path for output new weight file
    n_source: <int>
        Number of grid cells in the source grid
    n_target: <int>
        Number of grid cells in the target grid
    da_source_domain: <xr.DataArray>
        Domain file for the source domain. Should be 0 or 1 mask.
    process_method:
        This option is not implemented yet (right now, there is only one way of processing
        the weight file). Can be extended to have the option of, e.g., setting a threshold
        for whether to remap for a target cell or not if the coverage is low

    Requres
    ----------
    from scipy.sparse import coo_matrix
    import xesmf as xe
    '''

    # --- Read in the original xESMF weight file --- #
    A = xe.frontend.read_weights(orig_weight_nc, n_source, n_target)
    weight_array = A.toarray()  # [n_target, n_source]

    # --- For grid cells in the source domain that is inactive, assign weight 0 --- #
    # --- (xESMF always assumes full rectangular domain and does not consider domain shape) --- #
    # Flatten the source domain file
    N_source_lat = da_source_domain.values.shape[0]
    N_source_lon = da_source_domain.values.shape[1]
    source_domain_flat = da_source_domain.values.reshape([N_source_lat * N_source_lon])
    # Set weights for the masked source domain as zero
    masked_flag_flat = (source_domain_flat > -10e-15) & (source_domain_flat < 10e-15)
    weight_array[:, masked_flag_flat] = 0

    # --- Adjust weights for target grid cells whose sum < 1 --- #
    # Loop over each target grid cell
    for i in range(n_target):
        sum_weight = weight_array[i, :].sum()
        # If sum of weight is 0, there is no active source cell in the target cell.
        # Set all weights for this target cell to -1
        if sum_weight > -10e-14 and sum_weight < 10e-14:
            weight_array[i, :] = -1
        # Otherwise, the sum of weight should really be 1. If the sum < 1, rescale to 1
        elif sum_weight < (1 - 10e-14):
            weight_array[i, :] /= sum_weight
        elif sum_weight > (1 + 10e-14):
            raise ValueError('Error: xESMF weight sum exceeds 1. Something is wrong!')

    # --- Write new weights to file --- #
    data = weight_array[weight_array!=0]
    ind = np.where(weight_array!=0)
    row = ind[0] + 1  # adjust index to start from 1
    col = ind[1] + 1
    ds_weight_corrected = xr.Dataset({'S': (['n_s'],  data),
                                      'col': (['n_s'],  col),
                                      'row': (['n_s'],  row)},
                                     coords={'n_s': (['n_s'], range(len(data)))})
    ds_weight_corrected.to_netcdf(output_weight_nc, format='NETCDF4_CLASSIC')
    return weight_array


def calculate_crps(out_nc, ds_truth, ds_model, var, depth_sm=None):
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
            sm1; sm2; sm3
        NOTE: sm's and swe are from states; runoff's are from history file
    depth_sm: <xr.DataArray>
        Thickness of soil moisture
        Only required if state_var is soil moisture
        
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
        # --- Calculate CRPS for the whole domain --- #
        crps_domain = np.asarray(
            [crps(da_truth.sel(lat=lat, lon=lon).values,
                  da_model.sel(lat=lat, lon=lon).transpose('time', 'N').values)
             for lat in da_truth['lat'].values
             for lon in da_truth['lon'].values])
        # --- Reshape results --- #
        crps_domain = crps_domain.reshape([len(da_truth['lat']), len(da_truth['lon'])])
        # --- Put results into da's --- #
        da_crps = xr.DataArray(
            crps_domain, coords=[da_truth['lat'], da_truth['lon']],
            dims=['lat', 'lon'])
        # Save RMSE to netCDF file
        ds_crps = xr.Dataset(
            {'crps': da_crps})
        ds_crps.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
    else:  # if RMSE is already calculated
        da_crps = xr.open_dataset(out_nc)['crps']
    
    return da_crps


def calculate_bias_ensemble_norm_var(out_nc, ds_truth, ds_model, var, depth_sm=None):
    ''' A wrap funciton that calculates variance of ensemble-normalized bias for all domain
    and save to file; if result file already existed, then simply read in the file.

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
            sm1; sm2; sm3
        NOTE: sm's and swe are from states; runoff's are from history file
    depth_sm: <xr.DataArray>
        Thickness of soil moisture
        Only required if state_var is soil moisture
        
    Returns
    ----------
    da_bias_norm_var: <xr.DataArray>
        Variance of ensemble-normalized bias for the whole domain; dimension: [lat, lon]
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
        # --- Calculate bias_norm for the whole domain --- #
        bias_norm_var_domain = np.asarray(
            [bias_ensemble_norm_var(
                da_truth.sel(lat=lat, lon=lon).values,
                da_model.sel(lat=lat, lon=lon).transpose('time', 'N').values)
             for lat in da_truth['lat'].values
             for lon in da_truth['lon'].values])
        # --- Reshape results --- #
        bias_norm_var_domain = bias_norm_var_domain.reshape(
            [len(da_truth['lat']), len(da_truth['lon'])])
        # --- Put results into da's --- #
        da_bias_norm_var = xr.DataArray(
            bias_norm_var_domain, coords=[da_truth['lat'], da_truth['lon']],
            dims=['lat', 'lon'])
        # Save RMSE to netCDF file
        ds_bias_norm_var = xr.Dataset(
            {'bias_norm_var': da_bias_norm_var})
        ds_bias_norm_var.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
    else:  # if RMSE is already calculated
        da_bias_norm_var = xr.open_dataset(out_nc)['bias_norm_var']
    
    return da_bias_norm_var


def calculate_nensk(out_nc, ds_truth, ds_model, var, depth_sm=None):
    ''' A wrap funciton that calculates NENSK for all domain
    and save to file; if result file already existed, then simply read in the file.

    Parameters
    ----------
    out_nc: <str>
        RMSE result output netCDF file
    ds_truth: <xr.Dataset>
        Truth states/history
    ds_model: <xr.Dataset>
        Model states/history whose RMSE is to be assessed (wrt. truth states);
        This should be ensemble model results, with "N" as the ensemble dimension
        NOTE: this should already be daily data!!
    var: <str>
        Variable, options:
            sm1; sm2; sm3
        NOTE: sm's and swe are from states; runoff's are from history file
    depth_sm: <xr.DataArray>
        Thickness of soil moisture
        Only required if state_var is soil moisture
        
    Returns
    ----------
    da_bias_norm_var: <xr.DataArray>
        Variance of ensemble-normalized bias for the whole domain; dimension: [lat, lon]
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
        # --- Calculate nensk for the whole domain --- #
        nensk_domain = np.asarray(
            [nensk(
                da_truth.sel(lat=lat, lon=lon).values,
                da_model.sel(lat=lat, lon=lon).transpose('time', 'N').values)
             for lat in da_truth['lat'].values
             for lon in da_truth['lon'].values])
        # --- Reshape results --- #
        nensk_domain = nensk_domain.reshape(
            [len(da_truth['lat']), len(da_truth['lon'])])
        # --- Put results into da's --- #
        da_nensk = xr.DataArray(
            nensk_domain, coords=[da_truth['lat'], da_truth['lon']],
            dims=['lat', 'lon'])
        # Save RMSE to netCDF file
        ds_nensk = xr.Dataset(
            {'nensk': da_nensk})
        ds_nensk.to_netcdf(out_nc, format='NETCDF4_CLASSIC')
    else:  # if RMSE is already calculated
        da_nensk = xr.open_dataset(out_nc)['nensk']
    
    return da_nensk


def crps(truth, ensemble):
    ''' Calculate mean CRPS of an ensemble time series
    Parameters
    ----------
    truth: <np.array>
        A 1-D array of truth time series
        Dimension: [n]
    ensemble: <np.array>
        A 2-D array of ensemble time series
        Dimension: [n, N], where N is ensemble size; n is time series length
        
    Returns
    ----------
    crps: <float>
        Time-series-mean CRPS
        
    Require
    ----------
    import properscoring as ps
    '''
    
    array_crps = np.asarray([ps.crps_ensemble(truth[t], ensemble[t, :]) for t in range(len(truth))])
    crps = array_crps.mean()
    
    return crps


def bias_ensemble_norm_var(truth, ensemble):
    ''' Calculate variance of normalized bias of an ensemble time series.
    Specifically, at each time step t, mean bias is normalized by ensemble spread:
            bias_norm(t) = mean_bias / std(ensemble)
    Then average over all time steps:
            bias_norm = mean(bias_norm(t))
            
    Parameters
    ----------
    truth: <np.array>
        A 1-D array of truth time series
        Dimension: [n]
    ensemble: <np.array>
        A 2-D array of ensemble time series
        Dimension: [n, N], where N is ensemble size; n is time series length
        
    Returns
    ----------
    bias_ensemble_norm_var: <float>
        Time-series-mean ensemble-normalized bias
    '''
    
    mean_bias = ensemble.mean(axis=1) - truth  # [n]
    std_ensemble = ensemble.std(axis=1)  # [n]
    bias_ensemble_norm_var = (mean_bias / std_ensemble).var()
    
    return bias_ensemble_norm_var


def nensk(truth, ensemble):
    ''' Calculate the ratio of temporal-mean ensemble skill to temporal-mean ensemble spread:
            nensk = <ensk> / <ensp>
    where <ensk> is temporal average of: ensk(t) = (ensmean - truth)^2
          <ensp> is temperal average of: ensp(t) = mean((ens_i - ensmean)^2) = var(ens_i)
            
    Parameters
    ----------
    truth: <np.array>
        A 1-D array of truth time series
        Dimension: [n]
    ensemble: <np.array>
        A 2-D array of ensemble time series
        Dimension: [n, N], where N is ensemble size; n is time series length
        
    Returns
    ----------
    nensk: <float>
        Normalized ensemble skill
    '''
    
    ensk = np.square((ensemble.mean(axis=1) - truth))  # [n]
    ensp = ensemble.var(axis=1)  # [n]
    nensk = np.mean(ensk) / np.mean(ensp)
    
    return nensk


