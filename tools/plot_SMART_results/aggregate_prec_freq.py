
import xarray as xr
import sys
import os
import pandas as pd
from tonic.io import read_configobj

from da_utils import load_nc_and_concat_var_years

cfg = read_configobj(sys.argv[1])
prec_type = sys.argv[2]  # "corrected" or "perturbed"
freq = sys.argv[3]
i = int(sys.argv[4])


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

# --- Load subdaily prec file --- #
start_time = pd.to_datetime(cfg['SMART_RUN']['start_time'])
end_time = pd.to_datetime(cfg['SMART_RUN']['end_time'])
start_year = start_time.year
end_year = end_time.year
post_dir = os.path.join(
        cfg['CONTROL']['root_dir'], cfg['OUTPUT']['output_basedir'], 'post_SMART')
da_prec = load_nc_and_concat_var_years(
    basepath=os.path.join(post_dir, 'prec_{}.ens{}.'.format(prec_type, i)),
    start_year=start_year,
    end_year=end_year,
    dict_vars={'prec': 'prec_corrected'})\
        ['prec'].sel(time=slice(start_time, end_time))

# --- Aggregate to the specified frequency --- #
da_prec_agg = da_prec.resample(freq, dim='time', how='sum')

# --- Save aggregated precip to file --- #
ds_prec_agg = xr.Dataset({'PREC': da_prec_agg})
to_netcdf_history_file_compress(
    ds_prec_agg,
    os.path.join(post_dir,
        'prec_{}.ens{}.{}_{}.{}.nc'.format(prec_type, i, start_year, end_year, freq)))


