
''' This script runs VIC from initial state as openloop.

    Usage:
        $ python run_data_assim.py <config_file_EnKF> mpi_proc
'''

import sys
import numpy as np
import xarray as xr
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tonic.io import read_config, read_configobj
from da_utils import propagate_linear_model


# ============================================================ #
# Command line arguments
# ============================================================ #
# Read config file
cfg = read_configobj(sys.argv[1])

# ============================================================ #
# Process config file
# ============================================================ #
start_time = pd.to_datetime(cfg['LINEAR_MODEL']['start_time'])
end_time = pd.to_datetime(cfg['LINEAR_MODEL']['end_time'])

ds_domain = xr.open_dataset(os.path.join(cfg['CONTROL']['root_dir'],
                                         cfg['INPUT']['domain_nc']))
lat_coord = ds_domain['lat']
lon_coord = ds_domain['lon']

out_state_basepath = os.path.join(cfg['CONTROL']['root_dir'],
                                  cfg['OUTPUT']['out_state_basepath'])
out_history_dir = os.path.join(cfg['CONTROL']['root_dir'],
                               cfg['OUTPUT']['out_history_dir'])
forcing_basepath = os.path.join(
                                cfg['CONTROL']['root_dir'],
                                cfg['INPUT']['forcing_basepath'])

dict_linear_model_param = {'r1': cfg['LINEAR_MODEL']['r1'],
                           'r2': cfg['LINEAR_MODEL']['r2'],
                           'r3': cfg['LINEAR_MODEL']['r3'],
                           'r12': cfg['LINEAR_MODEL']['r12'],
                           'r23': cfg['LINEAR_MODEL']['r23']}

# ============================================================ #
# Run linear model
# ============================================================ #
print('Running linear model...')
propagate_linear_model(
    start_time=start_time, end_time=end_time,
    lat_coord=lat_coord, lon_coord=lon_coord,
    model_steps_per_day=cfg['LINEAR_MODEL']['model_steps_per_day'],
    init_state_nc=None,
    out_state_basepath=out_state_basepath,
    out_history_dir=out_history_dir,
    out_history_fileprefix=cfg['OUTPUT']['out_history_fileprefix'],
    forcing_basepath=forcing_basepath,
    prec_varname=cfg['INPUT']['prec_varname'],
    dict_linear_model_param=dict_linear_model_param)


