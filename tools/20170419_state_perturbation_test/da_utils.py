
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

from tonic.models.vic.vic import VIC, default_vic_valgrind_error_code

import timeit


class VICReturnCodeError(Exception):
    pass

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


def propagate(start_time, end_time, vic_exe, vic_global_template_file,
                       vic_model_steps_per_day, init_state_nc, out_state_basepath,
                       out_history_dir, out_history_fileprefix,
                       out_global_basepath, out_log_dir,
                       forcing_basepath, mpi_proc=None, mpi_exe='mpiexec'):
    ''' This function propagates (via VIC) from an initial state (or no initial state)
        to a certain time point.

    Parameters
    ----------
    start_time: <pandas.tslib.Timestamp>
        Start time of this propagation run
    end_time: <pandas.tslib.Timestamp>
        End time of this propagation
    vic_exe: <class 'VIC'>
        Tonic VIC class
    vic_global_template_file: <str>
        Path of VIC global file template
    vic_model_steps_per_day: <str>
        VIC option - model steps per day
    init_state_nc: <str>
        Initial state netCDF file; None for no initial state
    out_state_basepath: <str>
        Basepath of output states; ".YYYYMMDD_SSSSS.nc" will be appended.
        None if do not want to output state file
    out_history_dir: <str>
        Directory of output history files
    out_history_fileprefix: <str>
        History file prefix
    out_global_basepath: <str>
        Basepath of output global files; "YYYYMMDD-HHS_YYYYMMDD.txt" will be appended
    out_log_dir: <str>
        Directory for output log files
    forcing_basepath: <str>
        Forcing basepath. <YYYY.nc> will be appended
    mpi_proc: <int or None>
        Number of processors to use for VIC MPI run. None for not using MPI
        Default: None
    mpi_exe: <str>
        Path for MPI exe. Only used if mpi_proc is not None


    Require
    ----------
    OrderedDict
    generate_VIC_global_file
    check_returncode
    '''

    # Generate VIC global param file
    replace = OrderedDict([('FORCING1', forcing_basepath),
                           ('OUTFILE', out_history_fileprefix)])
    global_file = generate_VIC_global_file(
                        global_template_path=vic_global_template_file,
                        model_steps_per_day=vic_model_steps_per_day,
                        start_time=start_time,
                        end_time=end_time,
                        init_state='#INIT_STATE' if init_state_nc is None
                                   else 'INIT_STATE {}'.format(init_state_nc),
                        vic_state_basepath=out_state_basepath,
                        vic_history_file_dir=out_history_dir,
                        replace=replace,
                        output_global_basepath=out_global_basepath)

    # Run VIC
    returncode = vic_exe.run(global_file, logdir=out_log_dir,
                             **{'mpi_proc': mpi_proc, 'mpi_exe': mpi_exe})
    check_returncode(returncode, expected=0)

#    # Delete log files (to save space)
#    for f in glob.glob(os.path.join(out_log_dir, "*")):
#        os.remove(f)


def generate_VIC_global_file(global_template_path, model_steps_per_day,
                             start_time, end_time, init_state, vic_state_basepath,
                             vic_history_file_dir, replace,
                             output_global_basepath):
    ''' This function generates a VIC global file from a template file.

    Parameters
    ----------
    global_template_path: <str>
        VIC global parameter template (some parts to be filled in)
    model_steps_per_day: <int>
        VIC model steps per day for model run, runoff run and output
    start_time: <pandas.tslib.Timestamp>
        Model run start time
    end_time: <pandas.tslib.Timestamp>
        Model run end time (the beginning of the last time step)
    init_state: <str>
        A full line of initial state option in the global file.
        E.g., "# INIT_STATE"  for no initial state;
              or "INIT_STATE /path/filename" for an initial state file
    vic_state_basepath: <str>
        Output state name directory and file name prefix.
        None if do not want to output state file.
    vic_history_file_dir: <str>
        Output history file directory
    replace: <collections.OrderedDict>
        An ordered dictionary of globap parameters to be replaced
    output_global_basepath: <str>
        Output global file basepath
        ".<start_time>_<end_date>.nc" will be appended,
            where <start_time> is in '%Y%m%d-%H%S',
                  <end_date> is in '%Y%m%d' (since VIC always runs until the end of a date)

    Returns
    ----------
    output_global_file: <str>
        VIC global file path

    Require
    ----------
    string
    OrderedDict
    pandas
    '''

    # --- Create template string --- #
    with open(global_template_path, 'r') as global_file:
        global_param = global_file.read()

    s = string.Template(global_param)

    # --- Fill in global parameter options --- #
    state_time = end_time + pd.DateOffset(days=1/model_steps_per_day)

    global_param = s.safe_substitute(model_steps_per_day=model_steps_per_day,
                                     startyear=start_time.year,
                                     startmonth=start_time.month,
                                     startday=start_time.day,
                                     startsec=start_time.hour*3600+start_time.second,
                                     endyear=end_time.year,
                                     endmonth=end_time.month,
                                     endday=end_time.day,
                                     init_state=init_state,
                                     statename=vic_state_basepath,
                                     stateyear=state_time.year, # save state at the end of end_time time step (end_time is the beginning of that time step)
                                     statemonth=state_time.month,
                                     stateday=state_time.day,
                                     statesec=state_time.hour*3600+state_time.second,
                                     result_dir=vic_history_file_dir)

    # --- Replace global parameters in replace --- #
    global_param = replace_global_values(global_param, replace)

    # --- If vic_state_basepath == None, add "#" in front of STATENAME --- #
    if vic_state_basepath is None:
        for i, line in enumerate(global_param):
            if line.split()[0] == 'STATENAME':
                global_param[i] = "# STATENAME"

    # --- Write global parameter file --- #
    output_global_file = '{}.{}_{}.txt'.format(
                                output_global_basepath,
                                start_time.strftime('%Y%m%d-%H%S'),
                                end_time.strftime('%Y%m%d'))

    with open(output_global_file, mode='w') as f:
        for line in global_param:
            f.write(line)

    return output_global_file


def replace_global_values(gp, replace):
    '''given a multiline string that represents a VIC global parameter file,
       loop through the string, replacing values with those found in the
       replace dictionary'''

    gpl = []
    for line in iter(gp.splitlines()):
        line_list = line.split()
        if line_list:
            key = line_list[0]
            if key in replace:
                value = replace.pop(key)
                val = list([str(value)])
            else:
                val = line_list[1:]
            gpl.append('{0: <20} {1}\n'.format(key, ' '.join(val)))

    if replace:
        for key, val in replace.items():
            try:
                value = ' '.join(val)
            except:
                value = val
            gpl.append('{0: <20} {1}\n'.format(key, value))

    return gpl


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


def modify_sm_states(state_orig_nc, sm_pert, da_max_moist, out_state_nc):
    ''' Modify a prescibed amount of SM states for a single VIC state file.
        Perturbation amount can differ for each layer, but is the same (in mm)
        for the whole domain

    Parameters
    ----------
    state_orig_nc: <str>
        Original VIC state nc
    sm_pert: <list>
        A list of perturbation amount (in mm) for each layer
    da_max_moist: <xr.DataArray>
        Maximum soil moisture for each tile [nlyaer, lat, lon]
    out_state_nc: <str>
        Output state nc
    '''

    # Load original state file
    ds_state_orig = xr.open_dataset(state_orig_nc)
    # Add perturbation
    ds_state_perturbed = ds_state_orig.copy()
    for l in range(len(sm_pert)):
        ds_state_perturbed['STATE_SOIL_MOISTURE'][:, :, l, :, :] += sm_pert[l]
        # Limit perturbation to be between zero and upper bound
        for lat in da_max_moist['lat'].values:
            for lon in da_max_moist['lon'].values:
                sm = ds_state_perturbed['STATE_SOIL_MOISTURE']\
                        .loc[:, :, l, lat, lon].values
                # Set negative to zero
                sm[sm<0] = 0
                # Set above-maximum to maximum
                max_moist = da_max_moist.sel(lat=lat, lon=lon, nlayer=l).values
                sm[sm>max_moist] = max_moist
                # Put back into state ds
                ds_state_perturbed['STATE_SOIL_MOISTURE']\
                    .loc[:, :, l, lat, lon] = sm
    # Save perturbed states to file
    ds_state_perturbed.to_netcdf(out_state_nc, format='NETCDF4_CLASSIC')




