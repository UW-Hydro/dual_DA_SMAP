
import numpy as np
import pandas as pd
import os
import xarray as xr
import xesmf as xe


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

