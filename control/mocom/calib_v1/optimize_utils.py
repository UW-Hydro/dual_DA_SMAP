
import numpy as np
import datetime as dt
import pandas as pd
import xarray as xr


def read_RVIC_output(filepath, output_format='array', outlet_ind=-1):
    ''' This function reads RVIC output netCDF file

    Input:
        filepath: path of the output netCDF file
        output_format: 'array' or 'grid' (currently only support 'array')
        outlet_ind: index of the outlet to be read (index starts from 0); -1 for reading all outlets

    Return:
        df - a DataFrame containing streamflow [unit: cfs]; column name(s): outlet name
        dict_outlet - a dictionary with outlet name as keys; [lat lon] as content

    '''
    
    ds = xr.open_dataset(filepath)

    #=== Read in outlet names ===#
    outlet_names = [outlet_name.decode('utf-8')
                    for outlet_name in ds['outlet_name'].values]

    #=== Read in outlet lat lon ===#
    dict_outlet = {}
    # If read all outlets
    if outlet_ind==-1:
        for i, name in enumerate(outlet_names):
            dict_outlet[name] = [ds['lat'].values[i], ds['lon'].values[i]]
    # If read one outlet
    else:
        dict_outlet[outlet_names[outlet_ind]] = \
                        [ds['lat'].values[outlet_ind], ds['lon'].values[outlet_ind]]

    #=== Read in streamflow variable ===#
    flow = ds['streamflow'].values
    flow = flow * np.power(1000/25.4/12, 3)  # convert m3/s to cfs
    # If read all outlets
    if outlet_ind==-1:
        df = pd.DataFrame(flow, index=ds.coords['time'].values, columns=outlet_names)
    # If read one outlet
    else:
        df = pd.DataFrame(flow[:,outlet_ind], index=ds.coords['time'].values, \
                          columns=[outlet_names[outlet_ind]])

    return df, dict_outlet


def read_USGS_data(file, columns, names):
    '''This function reads USGS streamflow from the directly downloaded format (date are in the 3rd columns)

    Input:
        file: directly downloaded streamflow file path [str]
        columns: a list of data colomn numbers, starting from 1.
            E.g., if the USGS original data has three variables: max_flow, min_flow,
            mean_flow, and the desired variable is mean_flow, then columns = [3]
        names: a list of data column names. E.g., ['mean_flow']; must the same length as columns

    Return:
        a pd.DataFrame object with time as index and data columns (NaN for missing data points)

    Note: returned data and flow might not be continuous if there is missing data!!!

    '''
    ndata = len(columns)
    if ndata != len(names):  # check input validity
        raise ValueError("Input arguments 'columns' and 'names' must have same length!")

    f = open(file, 'r')
    date_array = []
    data = []
    for i in range(ndata):
        data.append([])
    while 1:
        line = f.readline().rstrip("\n")  # read in one line
        if line=="":
                break
        line_split = line.split('\t')
        if line_split[0]=='USGS':  # if data line
                date_string = line_split[2]  # read in date string
                date = dt.datetime.strptime(date_string, "%Y-%m-%d")  # convert date to dt object
                date_array.append(date)

                for i in range(ndata):  # for each desired data variable
                        col = columns[i]
                        if line_split[3+(col-1)*2] == '':  # if data is missing
                                value = np.nan
                        elif line_split[3+(col-1)*2] == 'Ice':  # if data is 'Ice'
                                value = np.nan
                        else:  # if data is not missing
                                value = float(line_split[3+(col-1)*2])
                        data[i].append(value)

    data = np.asarray(data).transpose()
    df = pd.DataFrame(data, index=date_array, columns=names)
    return df


def kge(sim, obs):
    ''' Calculate Kling-Gupta Efficiency (function from Oriana) '''

    std_sim = np.std(sim)
    std_obs = np.std(obs)
    mean_sim = sim.mean(axis=0)
    mean_obs = obs.mean(axis=0)
    r_array = np.corrcoef(sim.values, obs.values)
    r = r_array[0,1]
    relvar = std_sim/std_obs
    bias = mean_sim/mean_obs
    kge = 1-np.sqrt(np.square(r-1) + np.square(relvar-1) + np.square(bias-1))
    return kge


def nse(sim, obs):
    ''' Calcualte Nash–Sutcliffe efficiency'''
    
    obs_mean = np.mean(obs)
    nse = 1 - np.sum(np.square(sim - obs)) / np.sum(np.square(obs - obs_mean))
    return nse


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


