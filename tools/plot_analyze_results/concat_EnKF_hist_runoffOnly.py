
import xarray as xr
import numpy as np
import sys
import os


# --- Parameters --- #
ens = sys.argv[1]
EnKF_result_basedir = '/civil/hydro/ymao/data_assim/output/EnKF/ArkRed/random1.sm1_5.sm2_1.R_9.N30.Maurer_param.bc'
start_year = 1980
end_year = 1989

# --- Process --- #
out_nc = os.path.join(
    EnKF_result_basedir, 'history', 'EnKF_ensemble_concat',
    'history.runoff.ens{}.concat.{}_{}.nc'.format(ens, start_year, end_year))
if os.path.isfile(out_nc):  # If already exists
    ds = xr.open_dataset(out_nc)
    list_ds_EnKF_hist_allEns.append(ds)
else:
    print('Concat ensemble {}'.format(ens))
    # Concat all years
    list_ds_allyears = []
    for year in range(start_year, end_year+1):
        print(year)
        ds = xr.open_dataset(os.path.join(
            EnKF_result_basedir, 'history', 'EnKF_ensemble_concat',
            'history.ens{}.concat.{}.nc'.format(ens, year)))
        da_runoff = ds['OUT_RUNOFF']
        da_baseflow = ds['OUT_BASEFLOW']
        list_ds_allyears.append(xr.Dataset({'OUT_RUNOFF': da_runoff,
                                            'OUT_BASEFLOW': da_baseflow}))
    print('Concat...')
    ds_allyears = xr.concat(list_ds_allyears, dim='time')
    print('Save...')
    # Save to file
    ds_allyears.to_netcdf(out_nc, format='NETCDF4_CLASSIC')

