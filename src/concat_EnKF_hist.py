
import xarray as xr
import sys
import os
import pandas as pd
from tonic.io import read_configobj

cfg = read_configobj(sys.argv[1])
i = int(sys.argv[2])


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


# --- Load all years --- #
start_year = pd.to_datetime(cfg['EnKF']['start_time']).year
end_year = pd.to_datetime(cfg['EnKF']['end_time']).year
hist_output_dir = os.path.join(cfg['CONTROL']['root_dir'], cfg['OUTPUT']['output_EnKF_basedir'],
                               'history', 'EnKF_ensemble_concat')

list_ds = []
for y in range(start_year, end_year+1):
    ds = xr.open_dataset(os.path.join(hist_output_dir,
                                      'history.ens{}.concat.{}.nc'.format(i, y)))
    list_ds.append(ds)
ds_allyears = xr.concat(list_ds, dim='time')
# --- Save to file --- #
to_netcdf_history_file_compress(
    ds_allyears,
    os.path.join(hist_output_dir,
        'history.ens{}.concat.{}_{}.nc'.format(i, start_year, end_year)))
# --- Clean up --- #
for y in range(start_year, end_year+1):
    os.remove(os.path.join(hist_output_dir, 'history.ens{}.concat.{}.nc'.format(i, y)))



