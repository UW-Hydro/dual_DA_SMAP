import numpy as np
import pandas as pd
import os
import string
from collections import OrderedDict
import xarray as xr
import datetime as dt
import multiprocessing as mp
import shutil
import scipy.linalg as la
import glob
import h5py
from scipy.stats import rankdata
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.sparse import coo_matrix
import xesmf as xe

from tonic.models.vic.vic import VIC, default_vic_valgrind_error_code

import timeit


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


def calculate_smap_domain_from_vic_domain(da_vic_domain, da_smap_example):
    ''' Calculate the smallest SMAP domain needed to cover the entire VIC domain.
        Note: here the entire input VIC domain is going to be covered without considering
        its 0/1 masks.
    
    Parameters
    ----------
    da_vic_domain: <xr.DataArray>
        Input VIC domain. Here the entire input VIC domain is going to be covered without
        considering its 0/1 masks.
    da_smap_example: <xr.DataArray>
        An example SMAP DataArray. This typically can be the extracted SMAP data for one day.
    
    Returns
    ----------
    da_smap_domain: <xr.DataArray>
        SMAP domain
    '''
    
    # --- Calculate the range of lat lon edges of the VIC domain --- #
    vic_lat_edges = edges_from_centers(da_vic_domain['lat'].values)
    vic_lon_edges = edges_from_centers(da_vic_domain['lon'].values)
    vic_domain_lat_range = np.sort(np.array([vic_lat_edges[0], vic_lat_edges[-1]]))
    vic_domain_lon_range = np.sort(np.array([vic_lon_edges[0], vic_lon_edges[-1]]))
    # --- Calculate the smallest SMAP lat lon range that completely contains the VIC domain --- #
    # (Note that SMAP lats are in descending order)
    lats_smap = da_smap_example['lat'].values
    lons_smap = da_smap_example['lon'].values
    # lat lower
    smap_lat_edges = edges_from_centers(lats_smap)
    smap_lat_lower_ind = len(smap_lat_edges) - np.searchsorted(np.sort(smap_lat_edges), vic_domain_lat_range[0]) - 1
    smap_lat_lower = lats_smap[smap_lat_lower_ind]
    # lat upper
    smap_lat_upper_ind = len(smap_lat_edges) - np.searchsorted(np.sort(smap_lat_edges), vic_domain_lat_range[1]) - 1
    smap_lat_upper = lats_smap[smap_lat_upper_ind]
    # lon lower
    smap_lon_edges = edges_from_centers(lons_smap)
    smap_lon_lower_ind = np.searchsorted(np.sort(smap_lon_edges), vic_domain_lon_range[0]) - 1
    smap_lon_lower = lons_smap[smap_lon_lower_ind]
    # lon upper
    smap_lon_upper_ind = np.searchsorted(np.sort(smap_lon_edges), vic_domain_lon_range[1]) - 1
    smap_lon_upper = lons_smap[smap_lon_upper_ind]
    # --- Construct the SMAP domain needed --- #
    da_smap_domain_example = da_smap_example[0, :, :].sel(
        lat=slice(smap_lat_upper+0.05, smap_lat_lower-0.05),
        lon=slice(smap_lon_lower-0.05, smap_lon_upper+0.05))
    mask = np.ones([len(da_smap_domain_example['lat']), len(da_smap_domain_example['lon'])]).astype('int')
    da_smap_domain = xr.DataArray(
        mask,
        coords=[da_smap_domain_example['lat'], da_smap_domain_example['lon']],
        dims=['lat', 'lon'])
    
    return da_smap_domain


def extract_smap_static_info(filename, orbit="AM"):
    ''' This function extracts lat, lon and missing value info from raw SMAP L3 HDF5 file.
    
    Parameters
    ----------
    filename: <str>
        File path of a SMAP L3 HDF5 file
    orbit: <str>
        "AM" or "PM". Static info should be same for either orbit. Default: "AM".
    
    Reterns
    -------
    lat: <numpy.array>
        A 1-D array of sorted latitudes of the domain (descending order)
    lon: <numpy.array>
        A 1-D array of sorted longitudes of the domain (ascending order)
    '''
    
    with h5py.File(filename, 'r') as f:
        # Extract data info
        if orbit == "AM":
            lats = f['Soil_Moisture_Retrieval_Data_AM'.format(orbit)]['latitude'].value
            lons = f['Soil_Moisture_Retrieval_Data_AM']['longitude'].value
            missing_value = f['Soil_Moisture_Retrieval_Data_AM']['soil_moisture'].attrs['_FillValue']
        else:
            lats = f['Soil_Moisture_Retrieval_Data_PM'.format(orbit)]['latitude_pm'].value
            lons = f['Soil_Moisture_Retrieval_Data_PM']['longitude_pm'].value
            missing_value = f['Soil_Moisture_Retrieval_Data_PM']['soil_moisture_pm'].attrs['_FillValue']
        # Mask missing lat and lon
        latlat_masked = np.ma.masked_equal(lats, missing_value)  # 2-D
        lonlon_masked = np.ma.masked_equal(lons, missing_value)  # 2-D
        # Convert to clean 1-D lat and lon
        lat_uniq = np.unique(latlat_masked)
        lat = np.sort(np.array(lat_uniq[~np.ma.getmask(lat_uniq)]), )[::-1]  # lat in descending order
        lon_uniq = np.unique(lonlon_masked)
        lon = np.sort(np.array(lon_uniq[~np.ma.getmask(lon_uniq)]))  # lon in ascending order
    
    return lat, lon


def extract_smap_sm(filename, orbit, qc_retrieval_flag=False):
    ''' This function extracts soil moisture values [cm3/cm3] from a raw SMAP L3 HDF5 file.
    
    Parameters
    ----------
    filename: <str>
        File path of a SMAP L3 HDF5 file
    orbit: <str>
        "AM" or "PM"
    qc_retrieval_flag: <bool>
        Whether remove retrievals with non-zero quality flag. Default: False (do not remove)
    
    Reterns
    -------
    sm: <numpy.array>
        A 2-D soil moisture data of the whole domain (with np.nan for missing value)
    '''
    
    with h5py.File(filename, 'r') as f:
        # Extract soil moisture data
        if orbit == "AM":
            sm = f['Soil_Moisture_Retrieval_Data_AM']['soil_moisture'].value
            missing_value = f['Soil_Moisture_Retrieval_Data_AM']['soil_moisture'].attrs['_FillValue']
        else:
            sm = f['Soil_Moisture_Retrieval_Data_PM']['soil_moisture_pm'].value
            missing_value = f['Soil_Moisture_Retrieval_Data_PM']['soil_moisture_pm'].attrs['_FillValue']
        # Mask missing points
        sm[sm==missing_value] = np.nan
        # Remove non-zero retrieval quality flag, if specified
        if qc_retrieval_flag:
            if orbit == "AM":
                flags = f['Soil_Moisture_Retrieval_Data_AM']['retrieval_qual_flag'].value
            else:
                flags = f['Soil_Moisture_Retrieval_Data_PM']['retrieval_qual_flag_pm'].value
            sm[flags>0] = np.nan
        
    return sm


def extract_smap_multiple_days(filename, start_date, end_date, da_smap_domain=None):
    ''' This function imports a chunk of days of SMAP L3 data and put in a xr.DataArray, skipping missing dates

    Parameters
    ----------
    filename: <str>
        Template file path of SMAP L3 HDF5 files, with the "YYYYMMDD" part replaced by "{}", and the rest replaced by *.
        e.g.: './path/SMAP_L3_SM_P_{}_*.h5'
    start_date: <str>
        Start time of the period, in "YYYYMMDD" format.
        e.g.: "20150401"
    end_date: <str>
        End time of the period, in "YYYYMMDD" format.
        e.g.: "20150430"
    da_smap_domain: <xr.DataArray>
        SMAP domain to extract. None for keeping the global domain
    '''

    # Extract static info
    lat, lon = extract_smap_static_info(glob.glob(filename.format(start_date))[0])

    # Process time period
    dates = pd.date_range(start_date, end_date)  # dates only
    times = pd.date_range("{}-{:02d}".format(start_date, 6),
                          "{}-{:02d}".format(end_date, 18),
                          freq='12H')  # AM and PM measurement times

    # Initialize a DataArray for the entire global domain
    da_global = xr.DataArray(np.empty([len(lat), len(lon)]),
                             coords=[lat, lon], dims=['lat', 'lon'])

    # Initialize domain da
    if da_smap_domain is None:  # if there is no domain file, extract the entire global domain
        da = xr.DataArray(np.empty([len(times), len(lat), len(lon)]),
                          coords=[times, lat, lon], dims=['time', 'lat', 'lon'])
    if da_smap_domain is not None:  # if there is a domain file, extract domain only
        da = xr.DataArray(np.empty([len(times), len(da_smap_domain['lat']),
                                    len(da_smap_domain['lon'])]),
                          coords=[times, da_smap_domain['lat'], da_smap_domain['lon']],
                          dims=['time', 'lat', 'lon'])
    da[:] = np.nan

    # Load data for each day
    if da_smap_domain is not None:
        domain_lat_range = [da_smap_domain['lat'].values[0],
                            da_smap_domain['lat'].values[-1]]
        domain_lon_range = [da_smap_domain['lon'].values[0],
                            da_smap_domain['lon'].values[-1]]
    for date in dates:
        print('Loading {}'.format(date))
        date_str = date.strftime("%Y%m%d")  # date in YYYYMMMDD
        # --- Load AM data for this day --- #
        try:
            sm_am = extract_smap_sm(glob.glob(filename.format(date_str))[0], "AM", qc_retrieval_flag=True)
            if da_smap_domain is not None:
                da_global[:, :] = sm_am
                da_domain_data = da_global.sel(
                    lat=slice(domain_lat_range[0]+0.05, domain_lat_range[1]-0.05),
                    lon=slice(domain_lon_range[0]-0.05, domain_lon_range[1]+0.05))
                sm_am = da_domain_data.values
        except:
            print("Warning: cannot load AM data for {}. Assign missing value for this time.".format(date_str))
            continue
        # Put in the final da
        time = date + pd.DateOffset(hours=6)
        da.loc[time, :, :] = sm_am
        # --- Load PM data for this day --- #
        try:
            sm_pm = extract_smap_sm(glob.glob(filename.format(date_str))[0], "PM", qc_retrieval_flag=True)
            if da_smap_domain is not None:
                da_global[:, :] = sm_pm
                da_domain_data = da_global.sel(
                    lat=slice(domain_lat_range[0]+0.05, domain_lat_range[1]-0.05),
                    lon=slice(domain_lon_range[0]-0.05, domain_lon_range[1]+0.05))
                sm_pm = da_domain_data.values
        except:
            print("Warning: cannot load PM data for {}. Assign missing value for this time.".format(date_str))
            continue
        # Put in da
        time = date + pd.DateOffset(hours=18)
        da.loc[time, :, :] = sm_pm

    return da


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


def add_gridlines(axis, xlocs=[-80, -90, -100, -110, -120],
                  ylocs=[30, 35, 40], alpha=1):
    gl = axis.gridlines(draw_labels=True, xlocs=xlocs, ylocs=ylocs,
                        alpha=alpha)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return gl


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

    Return
    ----------
    da_remapped: <xr.DataArray>
        Remapped data

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


def rescale_SMAP_domain(da_smap, da_reference, smap_times_am, smap_times_pm,
                        da_meas_error_unscaled=None,
                        method='moment_2nd'):
    ''' Rescale SMAP data to be in the same regime of a reference field.
        Rescale each grid cell separately.
        AM and PM will be rescaled separately.
        Currently ignores all NANs in da_reference.
    
    Parameters
    ----------
    da_smap: <xr.DataArray>
        Original SMAP field. Dimension: [time, lat, lon]
    da_reference: <xr.DataArray>
        Refererence data. Dimension: [time, lat, lon].
        Can be different time length from da_input, but must be the same spatial grid.
    smap_times_am: <numpy.ndarray>
        Time points of SMAP AM
    smap_times_pm: <numpy.ndarray>
        Time points of SMAP PM
    da_meas_error_unscaled: <xr.DataArray>
        Unscaled domain of SMAP measurement error. Will be rescaled the same way as da_smap
        Dimension: [time, lat, lon]
        Default: None
    method: <str>
        Options: "moment_2nd" - matching mean and standard deviation
    
    Returns
    ----------
    da_smap_rescaled: <xr.DataArray>
        Rescaled SMAP data
    da_meas_error_rescaled: <xr.DataArray>
        Rescaled measurement error
    '''
    
    # --- Extract AM and PM data from da_reference
    # Extract SMAP AM time points
    da_reference_AMtimes = da_reference.sel(time=smap_times_am)
    # Extract SMAP PM time points
    da_reference_PMtimes = da_reference.sel(time=smap_times_pm)
    
    # --- Rescale SMAP data (for AM and PM seperately) --- #
    ncells = len(da_smap['lat']) * len(da_smap['lon'])
    # Rescale SMAP AM
    da_smap_AM = da_smap.sel(time=smap_times_am)
    da_meas_error_AM = da_meas_error_unscaled.sel(time=smap_times_am)
    da_smap_AM_rescaled, da_meas_error_AM_rescaled = rescale_domain(
        da_smap_AM, da_reference_AMtimes,
        da_meas_error_unscaled=da_meas_error_AM, method=method)
    # Rescale SMAP PM
    da_smap_PM = da_smap.sel(time=smap_times_pm)
    da_meas_error_PM = da_meas_error_unscaled.sel(time=smap_times_pm)
    da_smap_PM_rescaled, da_meas_error_PM_rescaled = rescale_domain(
        da_smap_PM, da_reference_PMtimes,
        da_meas_error_unscaled=da_meas_error_PM, method=method)
    # Put AM and PM back together
    da_smap_rescaled = xr.concat([da_smap_AM_rescaled, da_smap_PM_rescaled], dim='time').sortby('time')
    da_meas_error_rescaled = xr.concat([da_meas_error_AM_rescaled, da_meas_error_PM_rescaled], dim='time').sortby('time')

    return da_smap_rescaled, da_meas_error_rescaled


def rescale_domain(da_input, da_reference, da_meas_error_unscaled, method):
    ''' Rescales an input domain of time series to be in the same regime of a reference domain.
    Currently ignores all NANs in da_reference.
    
    Parameters
    ----------
    da_input: <xr.DataArray>
        Input data. Dimension: [time, lat, lon]
    da_reference: <xr.DataArray>
        Refererence data. Dimension: [time, lat, lon]. Can be different length from da_input.
    da_meas_error_unscaled: <xr.DataArray>
        Unscaled domain of SMAP measurement error. Will be rescaled the same way as da_smap
        Default: None
    method: <str>
        Options: "moment_2nd" - matching mean and standard deviation
    
    Returns
    ----------
    da_rescaled: <xr.DataArray>
        Rescaled data
    da_meas_error_scaled: <xr.DataArray.
        Rescaled measurement error
    '''
    
    # Rescale input data for each grid cell
    ncells = len(da_input['lat']) * len(da_input['lon'])
    da_input_flat = da_input.values.reshape([-1, ncells]) # [time, lat*lon]
    da_reference_flat = da_reference.values.reshape([-1, ncells])  # [time, lat*lon]
    list_data_rescaled_flat = np.asarray(
        [rescale_ts(pd.Series(da_input_flat[:, i], index=da_input['time'].values),
                    pd.Series(da_reference_flat[:, i], index=da_reference['time'].values),
                    method=method)
         for i in range(ncells)])  # [lat*lon, time]
    # Extract flattened scaled data, std_input and std_reference
    data_rescaled_flat = np.asarray([item[0] for item in list_data_rescaled_flat])
    std_input_rescaled_flat = np.asarray([item[1] for item in list_data_rescaled_flat])
    std_reference_rescaled_flat = np.asarray([item[2] for item in list_data_rescaled_flat])
    # Reshape
    data_rescaled = data_rescaled_flat.reshape(
        [len(da_input['lat']), len(da_input['lon']), -1])  # [lat, lon, time]
    data_rescaled = np.rollaxis(data_rescaled, 2, 0)  # [time, lat, lon]
    std_input_rescaled = std_input_rescaled_flat.reshape(
        [len(da_input['lat']), len(da_input['lon'])])  # [lat, lon]
    std_reference_rescaled = std_reference_rescaled_flat.reshape(
        [len(da_input['lat']), len(da_input['lon'])])  # [lat, lon]
    # Put back into da
    da_rescaled = da_input.copy()
    da_rescaled[:] = data_rescaled
    # Rescale measurement error
    da_meas_error_scaled = da_meas_error_unscaled.copy()
    da_meas_error_scaled[:] = da_meas_error_unscaled / std_input_rescaled * std_reference_rescaled
    
    return da_rescaled, da_meas_error_scaled


def rescale_ts(ts_input, ts_reference, method):
    ''' Rescales an input time series to be in the same regime of a reference time series.
    Currently ignores all NANs in ts_reference.

    Parameters
    ----------
    ts_input: <pd.Series>
        Input time series
    ts_reference: <pd.Series>
        Reference time series
    method: <str>
        Options: "moment_2nd" - matching mean and standard deviation
                 "moment_2nd_season" - matching mean and standard deviation; mean is
                     sampled using  31-day window of year; standard deviation is kept
                     constant temporally (following SMART paper 2011)
                 "cdf" - cdf matching. Will return None for ts_std_input and ts_std_reference

    Returns
    ----------
    ts_rescaled: <pd.Series>
        Rescaled time series
    std_input: <float>
        Input standard deviation
    std_reference: <float>
        Reference standard deviation
    '''

    if method == "moment_2nd":
        mean_reference = np.nanmean(ts_reference)
        std_reference = np.nanstd(ts_reference, ddof=0)
        mean_input = np.nanmean(ts_input)
        std_input = np.nanstd(ts_input, ddof=0)
        ts_rescaled = mean_reference + (ts_input - mean_input) / std_input * std_reference
        # Construct time series of ts_std_input and ts_std_reference to return
        ts_std_input = pd.Series(np.ones(len(ts_input))*std_input, index=ts_input.index)
        ts_std_reference = pd.Series(np.ones(len(ts_reference))*std_reference, index=ts_reference.index)
        
    elif method == "moment_2nd_season":
        std_reference = np.nanstd(ts_reference, ddof=0)
        std_input = np.nanstd(ts_input, ddof=0)
        # Calculate window-mean for both reference and input series
        # Dict key: (month, day); value: mean value from the ts
        dict_window_mean_reference = calculate_seasonal_window_mean(ts_reference)
        dict_window_mean_input = calculate_seasonal_window_mean(ts_input)
        # Construct time series of "mean_reference" and "mean_input"
        # (mean_reference ts is constructed at the input data timestep)
        list_mean_reference = [dict_window_mean_reference[(t.month, t.day)]
                               for t in ts_input.index]
        ts_mean_reference = pd.Series(list_mean_reference, index=ts_input.index)
        list_mean_input = [dict_window_mean_input[(t.month, t.day)]
                               for t in ts_input.index]
        ts_mean_input = pd.Series(list_mean_input, index=ts_input.index)
        # Rescale input ts
        ts_rescaled = ts_mean_reference + (ts_input - ts_mean_input) / std_input * std_reference

    elif method == "cdf":
        dict_window_data_input = extract_seasonal_window_data(ts_input)
        dict_window_data_reference = extract_seasonal_window_data(ts_reference)
        array_rescaled = np.array(
            [scipy.stats.mstats.mquantiles(
                dict_window_data_reference[(t.month, t.day)],
                calculate_ecdf_percentile(ts_input[t], dict_window_data_input[(t.month, t.day)]),
                alphap=0, betap=0)
             if ~np.isnan(ts_input[t]) else np.nan
             for t in ts_input.index ])
        ts_rescaled = pd.Series(array_rescaled, ts_input.index)
        ts_std_input = None
        ts_std_reference = None

    return ts_rescaled, std_input, std_reference


def calculate_seasonal_window_mean(ts):
    ''' Calculates seasonal window mean values of a time series.
        (mean value of 31-day-window of all years)

    Parameters
    ----------
    ts: <pd.Series>
        Time series to calculate

    Returns
    ----------
    dict_window_mean: <dict>
        Dict of window-mean values for the ts
        key: (month, day); value: mean value from the ts
    '''

    # Extract all indices for each (month, day) of a full year
    d_fullyear = pd.date_range('20160101', '20161231')
    dayofyear_fullyear = [(d.month, d.day) for d in d_fullyear]
    list_dayofyear_index = [(ts.index.month==d[0]) & (ts.index.day==d[1]) for d in dayofyear_fullyear]
    keys = dayofyear_fullyear
    values = list_dayofyear_index
    dict_dayofyear_index = dict(zip(keys, values))

    # Calculate window-mean value for each (month, day)
    dict_window_mean = {}  # key: (month, day); value: mean value from the ts
    for d in d_fullyear:
        # Identify (month, day)s in a 31-day window centered around the current day
        d_window = pd.date_range(d.date() - pd.DateOffset(days=15),
                                 d.date() + pd.DateOffset(days=15))
        dayofyear_window = [(d.month, d.day) for d in d_window]
        # Extract all data points in the window of all years
        ts_window = pd.concat([ts.loc[dict_dayofyear_index[d]]
                               for d in dayofyear_window])
        # Calculate window mean value
        mean_ts = ts_window.mean()
        dict_window_mean[(d.month, d.day)] = mean_ts

    return dict_window_mean


def load_nc_and_concat_var_years(basepath, start_year, end_year, dict_vars):
    ''' Loads in netCDF files end with 'YYYY.nc', and for each variable needed,
        concat all years together and return a DataArray

        Parameters
        ----------
        basepath: <str>
            Basepath of all netCDF files; 'YYYY.nc' will be appended;
            Time dimension name in the nc files must be 'time'
        start_year: <int>
            First year to load
        end_year: <int>
            Last year to load
        dict_vars: <dict>
            A dict of desired variables and corresponding varname in the
            netCDF files (e.g., {'prec': 'prcp'; 'temp': 'tair'}). The keys in
            dict_vars will be used as keys in the output dict.

        Returns
        ----------
        dict_da: <dict>
            A dict of concatenated xr.DataArrays.
            Keys: desired variables (using the same keys as in input
                  'dict_vars')
            Elements: <xr.DataArray>
    '''

    dict_list = {}
    for var in dict_vars.keys():
        dict_list[var] = []

    # Loop over each year
    for year in range(start_year, end_year+1):
        # Load data for this year
        ds = xr.open_dataset(basepath + '{}.nc'.format(year))
        # Extract each variable needed and put in a list
        for var, varname in dict_vars.items():
            da = ds[varname]
            # Put data of this year in a list
            dict_list[var].append(da)

    # Concat all years for all variables
    dict_da = {}
    for var in dict_vars.keys():
        dict_da[var] = xr.concat(dict_list[var], dim='time')

    return dict_da



