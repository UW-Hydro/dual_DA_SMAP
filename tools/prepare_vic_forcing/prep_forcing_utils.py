
import pandas as pd
import os
import numpy as np
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from collections import OrderedDict
import xesmf as xe
from scipy.sparse import coo_matrix


def to_netcdf_forcing_file_compress(ds_force, out_nc, time_dim='time'):
    ''' This function saves a VIC-forcing-file-format ds to netCDF, with
        compression.

    Parameters
    ----------
    ds_force: <xr.Dataset>
        Forcing dataset to save
    out_nc: <str>
        Path of output netCDF file
    time_dim: <str>
        Time dimension name in ds_force. Default: 'time'
    '''

    dict_encode = {}
    for var in ds_force.data_vars:
        # determine chunksizes
        chunksizes = []
        for i, dim in enumerate(ds_force[var].dims):
            if dim == time_dim:  # for time dimension, chunksize = 1
                chunksizes.append(1)
            else:
                chunksizes.append(len(ds_force[dim]))
        # create encoding dict
        dict_encode[var] = {'zlib': True,
                            'complevel': 1,
                            'chunksizes': chunksizes}
    ds_force.to_netcdf(out_nc,
                      format='NETCDF4',
                      encoding=dict_encode)


def calculate_gpm_domain_from_vic_domain(da_vic_domain, da_smap_example):
    ''' Calculate the smallest GPM domain needed to cover the entire VIC domain.
        Note: here the entire input VIC domain is going to be covered without considering
        its 0/1 masks.
        NOTE: this function is directly adapted from calculate_smap_domain_from_vic_domain;
        everything is the same except that SMAP has descending LAT but GPM has
        ascending LAT

    Parameters
    ----------
    da_vic_domain: <xr.DataArray>
        Input VIC domain. Here the entire input VIC domain is going to be covered without
        considering its 0/1 masks.
    da_smap_example: <xr.DataArray>
        An example GPM DataArray. This typically can be the extracted GPM data for one day.

    Returns
    ----------
    da_gpm_domain: <xr.DataArray>
        GPM domain
    '''

    # --- Calculate the range of lat lon edges of the VIC domain --- #
    vic_lat_edges = edges_from_centers(da_vic_domain['lat'].values)
    vic_lon_edges = edges_from_centers(da_vic_domain['lon'].values)
    vic_domain_lat_range = np.sort(np.array([vic_lat_edges[0], vic_lat_edges[-1]]))
    vic_domain_lon_range = np.sort(np.array([vic_lon_edges[0], vic_lon_edges[-1]]))
    # --- Calculate the smallest GPM lat lon range that completely contains the VIC domain --- #
    # (Note that GPM lats are in descending order)
    lats_smap = da_smap_example['lat'].values
    lons_smap = da_smap_example['lon'].values
    # lat lower
    smap_lat_edges = edges_from_centers(lats_smap)
    smap_lat_lower_ind = np.searchsorted(np.sort(smap_lat_edges),
                                         vic_domain_lat_range[0]) - 1
    smap_lat_lower = lats_smap[smap_lat_lower_ind]
    # lat upper
    smap_lat_upper_ind = np.searchsorted(np.sort(smap_lat_edges),
                                         vic_domain_lat_range[1]) - 1
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
        lat=slice(smap_lat_lower-0.05, smap_lat_upper+0.05),
        lon=slice(smap_lon_lower-0.05, smap_lon_upper+0.05))
    mask = np.ones([len(da_smap_domain_example['lat']),
                    len(da_smap_domain_example['lon'])]).astype('int')
    da_smap_domain = xr.DataArray(
        mask,
        coords=[da_smap_domain_example['lat'], da_smap_domain_example['lon']],
        dims=['lat', 'lon'])

    return da_smap_domain


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
    print('Remapping...')
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
        elif sum_weight < (1 - 10e-10):
            weight_array[i, :] /= sum_weight
        elif sum_weight > (1 + 10e-10):
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


