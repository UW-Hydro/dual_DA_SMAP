
import numpy as np
import datetime as dt
import pandas as pd
import os
import xarray as xr
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.io import reset_output
import bokeh
import properscoring as ps
from scipy import stats


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
        
    Require
    ----------
    import properscoring as ps
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


def get_z_values(ens, obs):
    ''' Calculate cumulative probability corresponding to observed
        flow (F(obs)) from forecast ensemble CDF (F(x)).
        These are z_i in Laio and Tamea 2007 Fig. 2.
        https://doi.org/10.5194/hess-11-1267-2007
        Input: ens = ensemble forecast for a single day
                     numpy.array, num_ensemble_members x 1
               obs = observation for a single day, one value.
        Output: z = cumulative probability of observation based on
                    forecast ensemble, single value
    '''
#    # construct forecast CDF from ensemble weights
#    cdf, ens_sort = construct_cdf(ens)
#    # get z = F(obs) from forecast CDF
#    z = cdf_simulated(obs, cdf, ens_sort)

    if obs < min(ens):
        z = 0
    elif obs > max(ens):
        z = 1
    else:
        z = stats.percentileofscore(ens, obs, 'mean') / 100
    return z


def get_z_values_timeseries(ens_ts, obs_ts):
    ''' Calculate z values for a time series of ensemble
    Parameters
    ----------
    ens_ts: <np.array>
        An ensemble of timeseries
        dim: [N, t] where N is ensemble size and t is timesteps
    obs_ts: <np.array>
        Time series of observations
        dim: [t]

    Returns
    ----------
    z_alltimes: <np.array>
        Quantile of observation in ensemble at all time series
        dim: [t]
    '''

    if len(obs_ts) != ens_ts.shape[1]:
        raise ValueError('Ensemble and observed time series not the same length!')

    z_alltimes = \
        np.asarray(
            [get_z_values(ens_ts[:, t], obs_ts[t])
             for t in range(len(obs_ts))])

    return z_alltimes



def calc_reliability_bias(z_daily):
    ''' Calculate alpha-reliability following Renard et al. 2010
        (Eqn. 23) https://doi.org/11.1029/2009WR008328. See also
        Laio and Tamea 2007
        https://doi.org/10.5194/hess-11-1267-2007
        Inputs: z_daily = daily z = F(obs) values. These correspond to
                          "quantile of observed p-values" from
                          Renard et al. 2010 Fig. 3,
                          "observed p-values of x_t" from Renard et
                          al. 2010 Eqn. 23b, and z_i from Laio and
                          Tamea 2007 Fig. 2.
                          numpy.array 1 x num_days
        Outputs: alpha_reliability
                Range: -1 to 1; best-performance value = 0
                positive value indicates over-prediction
                negative value indicates under-prediction
    '''

    z_daily = np.sort(z_daily)
    n_sample = len(z_daily)
    # assign ranks to daily z = F(obs) values.
    # rank/# samples gives us the "theoretical quantile of U[0,1]"
    # from Renard et al. 2010 Fig. 3, "theoretical p-values of x_t"
    # from Renard et al. 2010 Eqn. 23b, and R_i/n from Laio and Tamea
    # 2007 Fig. 2.
    # Note: Renard et al. 2010 Fig.3 flipped the x and y axes from
    # Laio and Tamea 2007 Fig. 2.
    R = np.arange(0, n_sample) / n_sample
    # calculate alpha reliability index
    alpha_direction = - 2 * np.mean(z_daily - R)
    return alpha_direction


def calc_alpha_reliability(z_daily):
    ''' Calculate alpha-reliability following Renard et al. 2010
        (Eqn. 23) https://doi.org/11.1029/2009WR008328. See also
        Laio and Tamea 2007
        https://doi.org/10.5194/hess-11-1267-2007
        Inputs: z_daily = daily z = F(obs) values. These correspond to
                          "quantile of observed p-values" from
                          Renard et al. 2010 Fig. 3,
                          "observed p-values of x_t" from Renard et
                          al. 2010 Eqn. 23b, and z_i from Laio and
                          Tamea 2007 Fig. 2.
                          numpy.array 1 x num_days
        Outputs: alpha = alpha reliability index, single value
    '''

    z_daily = np.sort(z_daily)
    n_sample = len(z_daily)
    # assign ranks to daily z = F(obs) values.
    # rank/# samples gives us the "theoretical quantile of U[0,1]"
    # from Renard et al. 2010 Fig. 3, "theoretical p-values of x_t"
    # from Renard et al. 2010 Eqn. 23b, and R_i/n from Laio and Tamea
    # 2007 Fig. 2.
    # Note: Renard et al. 2010 Fig.3 flipped the x and y axes from
    # Laio and Tamea 2007 Fig. 2.
    R = np.arange(0, n_sample) / n_sample
    # calculate alpha reliability index
    alpha = 1 - 2 * np.mean(abs(z_daily - R))
    return alpha


def calc_kesi(z_alltimes):
    ''' Calculate kesi (fraction of observed timesteps within the
        ensemble range)

    Parameters
    ----------
    z_alltimes: <np.array>
        z values of all timesteps; dim: [time]

    Returns
    ----------
    kesi: <float>
        kesi
    '''

    kesi = 1 - ((z_alltimes==1).sum() + (z_alltimes==0).sum()) \
        / len(z_alltimes)

    return kesi


