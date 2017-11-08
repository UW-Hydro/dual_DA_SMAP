
import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from da_utils import (Forcings, perturb_forcings_ensemble, setup_output_dirs,
                      to_netcdf_forcing_file_compress, calculate_sm_noise_to_add_magnitude,
                      calculate_scale_n_whole_field, to_netcdf_state_file_compress,
                      calculate_max_soil_moist_domain, convert_max_moist_n_state)


def load_nc_file(nc_file, start_year, end_year):
    ''' Loads in nc files for all years.

    Parameters
    ----------
    nc_file: <str>
        netCDF file to load, with {} to be substituted by YYYY
    start_year: <int>
        Start year
    end_year: <int>
        End year

    Returns
    ----------
    ds_all_years: <xr.Dataset>
        Dataset of all years
    '''

    list_ds = []
    for year in range(start_year, end_year+1):
        # Load data
        fname = nc_file.format(year)
        ds = xr.open_dataset(fname)
        list_ds.append(ds)
        # Concat all years
        ds_all_years = xr.concat(list_ds, dim='time')

    return ds_all_years


def pert_prec_state_cell_ensemble(
    N, state_times, corrcoef,
    ds_force_orig, prec_std, out_forcing_basedir,
    states_orig, scale_n_nloop, out_state_basedir, da_max_moist_n):
    ''' Perturb precipitation to produce a ensemble of perturbed forcing.
        Only for one-grid-cell case.
    
    Parameters
    ----------
    N: <int>
        Ensemble size
    state_times: <pandas.tseries.index.DatetimeIndex>
        Time points of antecedent states. This needs to be daily at 00:00 for now.
        Forcings will be aggregated to the day immediately following these times.
    corrcoef: <float>
        Correlation coefficient between the (additive/multiplicative) noises added
        to states and forcings
    ds_force_orig: <xr.Dataset>
        Original VIC-format forcing
    prec_std: <float>
        Standard deviation of the precipitation perturbing multiplier
    out_forcing_basedir: <str>
        Base directory for output perturbed forcings;
        Subdirs "ens_<i>" will be created, where <i> is ensemble index, 1, ..., N
        File names will be: forc.YYYY.nc
        states_orig: <list>
        List of original VIC-format states
    states_orig: <list>
        List of original VIC-format states
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon = 1)
    out_state_basedir: <str>
        Base directory for output perturbed states;
        Subdirs "ens_<i>" will be created, where <i> is ensemble index, 1, ..., N
        File names will be: state.YYYY_SSSSS.nc
    da_max_moist_n: <xarray.DataArray>
            Maximum soil moisture for the whole domain and each tile
            [unit: mm]. Soil moistures above maximum after perturbation will
            be reset to maximum value.
            Dimension: [lat, lon, n]
    '''
    
    for ens in range(N):
        print('Perturbing forcing & states ens. {}...'.format(ens+1))
        multiplier_prec_daily, noise_states_scaled = pert_prec_state_cell(
            ens, state_times=state_times, corrcoef=corrcoef,
            ds_force_orig=ds_force_orig, prec_std=prec_std,
            out_forcing_basedir=out_forcing_basedir,
            states_orig=states_orig, scale_n_nloop=scale_n_nloop,
            out_state_basedir=out_state_basedir,
            da_max_moist_n=da_max_moist_n,
            seed=None)


def pert_prec_state_cell(ens, state_times, corrcoef,
                         ds_force_orig, prec_std, out_forcing_basedir,
                         states_orig, scale_n_nloop, out_state_basedir, da_max_moist_n,
                         seed=None):
    ''' Perturb precipitation to produce a perturbed forcing.
        Only for one-grid-cell case.

    Parameters
    ----------
    ens: <int>
        Index of ensemble for perturbation (start from 0)
    state_times: <pandas.tseries.index.DatetimeIndex>
        Time points of antecedent states. This needs to be daily at 00:00 for now.
        Forcings will be aggregated to the day immediately following these times.
    corrcoef: <float>
        Correlation coefficient between the (additive/multiplicative) noises added
        to states and forcings
    ds_force_orig: <xr.Dataset>
        Original VIC-format forcing
    prec_std: <float>
        Standard deviation of the precipitation perturbing multiplier
    out_forcing_basedir: <str>
        Base directory for output perturbed forcings;
        Subdirs "ens_<i>" will be created, where <i> is ensemble index, 1, ..., N
        File names will be: forc.YYYY.nc
        states_orig: <list>
        List of original VIC-format states
    states_orig: <list>
        List of original VIC-format states
    scale_n_nloop: <np.array>
        Standard deviation of noise to add for the whole field.
        Dimension: [nloop, n] (where nloop = lat * lon = 1)
    out_state_basedir: <str>
        Base directory for output perturbed states;
        Subdirs "ens_<i>" will be created, where <i> is ensemble index, 1, ..., N
        File names will be: state.YYYY_SSSSS.nc
    da_max_moist_n: <xarray.DataArray>
            Maximum soil moisture for the whole domain and each tile
            [unit: mm]. Soil moistures above maximum after perturbation will
            be reset to maximum value.
            Dimension: [lat, lon, n]
    seed: <int or None>
        Seed for random number generator; this seed will only be used locally
        in this function and will not affect the upper-level code.
        None for not re-assign seed in this function, but using the global seed)
        Default: None
    '''

    # === Generate correlated noises for states and forcings === #
    ### Step 1: Generate two series of correlated noises, both with
    # zero mean and unit variance
    mean = [0, 0]
    cov = [[1, corrcoef], [corrcoef, 1]]
    if seed is None:
        noise_correlated = np.random.multivariate_normal(mean, cov, len(state_times))
    else:
        rng = np.random.RandomState(seed)
        noise_correlated = rng.multivariate_normal(mean, cov, len(state_times))
    ### Step 2: Transform noise for precipitation
    # Calculate mu and sigma for the lognormal distribution
    # (here mu and sigma are mean and std of the underlying normal dist.)
    mu_prec = -0.5 * np.log(prec_std^2 + 1)
    sigma_prec = np.sqrt(np.log(prec_std^2 + 1))
    # Scale the normal noise
    prec_noise_normal = noise_correlated[:, 0] * sigma_prec + mu_prec
    # Transform to daily multiplier
    multiplier_prec_daily = np.exp(prec_noise_normal)
    ### Step 3: Transform noise for SM states
    # Scale noise for each layer (and uniform for each tile)
    noise_standard = noise_correlated[:, 1].reshape([len(state_times), 1])  # [time, 1]
    noise_scaled = np.dot(noise_standard, scale_n_nloop)  # [time, n]
    noise_scaled = noise_scaled.reshape(
        [len(state_times), len(states_orig[0]['nlayer']), len(states_orig[0]['veg_class']),
         len(states_orig[0]['snow_band'])])  # [time, nlayer, nveg, nsnow]
    noise_states_scaled = np.rollaxis(noise_scaled, 1, 4)  # [time, nveg, nsnow, nlayer]
    
    # === Perturb precip. === #
    # --- Aggregate orig. prec to daily --- #
    da_prec_orig = ds_force_orig['PREC'].squeeze()  # [time]
    da_prec_orig_daily = da_prec_orig.groupby('time.date').sum()  # [day]
    # --- Distribute daily multipliers to sub-daily --- #
    da_multiplier_prec_daily = da_prec_orig_daily.copy(deep=True)
    da_multiplier_prec_daily[:] = multiplier_prec_daily
    da_multiplier_prec = da_prec_orig.copy(deep=True)
    times = [pd.to_datetime(time).date() for time in da_multiplier_prec['time'].values]
    for date, item in da_multiplier_prec.groupby('time.date'):
        time_ind = [time == date for time in times]
        da_multiplier_prec[time_ind] = da_multiplier_prec_daily.loc[date]
    # --- Perturb sub-daily prec --- #
    da_prec_pert = da_prec_orig * da_multiplier_prec
    # --- Save perturbed prec. to file --- #
    # Replace perturbed prec. in the original forcing data
    ds_force_pert = ds_force_orig.copy(deep=True)
    ds_force_pert['PREC'][:, 0, 0] = da_prec_pert
    # Set up ensemble subdir
    subdir_name = 'ens_{}'.format(ens+1)
    force_pert_ens_basedir = setup_output_dirs(
        out_forcing_basedir,
        mkdirs=[subdir_name])[subdir_name]
    for year, ds in ds_force_pert.groupby('time.year'):
        to_netcdf_forcing_file_compress(
            ds, os.path.join(force_pert_ens_basedir, 'force.{}.nc'.format(year)))

    # === Perturb states === #
    # Set up ensemble subdir
    subdir_name = 'ens_{}'.format(ens+1)
    state_pert_ens_basedir = setup_output_dirs(
        out_state_basedir,
        mkdirs=[subdir_name])[subdir_name]
    # Add noise to the orig. states and save to file
    for i, time in enumerate(state_times):
        datetime = pd.to_datetime(time)
        # Replace perturbed states in the orig. state
        ds_states_pert = states_orig[i].copy(deep=True)
        ds_states_pert['STATE_SOIL_MOISTURE'][:, :, :, 0, 0] += noise_states_scaled[i, :, :, :]
        # Adjust negative SM to zero
        sm_new = ds_states_pert['STATE_SOIL_MOISTURE'].values
        sm_new[sm_new<0] = 0
        # Reset perturbed soil moistures above maximum to maximum
        max_moist_n = np.rollaxis(da_max_moist_n.values, 2, 0)  # [n, lat, lon]
        max_moist = max_moist_n.reshape(
            [len(states_orig[0]['nlayer']), len(states_orig[0]['veg_class']),
             len(states_orig[0]['snow_band']), 1, 1])  # [nlayer, nveg, nsnow, lat, lon]
        max_moist = np.rollaxis(max_moist, 0, 3)  # [nveg, nsnow, nlayer, lat, lon]
        sm_new[(sm_new>max_moist)] = max_moist[(sm_new>max_moist)]
        ds_states_pert['STATE_SOIL_MOISTURE'][:] = sm_new
        # Save
        to_netcdf_state_file_compress(
            ds_states_pert,
            os.path.join(state_pert_ens_basedir,
                         'state.{}_{:05d}.nc'.format(datetime.strftime('%Y%m%d'), datetime.second)))
    
    return multiplier_prec_daily, noise_states_scaled

