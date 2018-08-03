# Steps of running dual correction system

## Synthetic experiment:
1.  Generate ensemble precipitation (for state EnKF)

    `$ python gen_ensemble_forcing.py <cfg> <ens>`

2.  Generate synthetic truth and measurements

    `$ python gen_synthetic_meas.py <cfg> <mpi_proc>`

3.  Run DA for state updating

    `$ python run_data_assim.py <cfg> <nproc> <mpi_proc> <debug> <restart> < save_cellAvg_state_only>`
(set <save_cellAvg_state_only>=True if do not need to reuse the updated states to run VIC to save space)

If want to evaluate ensemble statistics, need to do a similar DA run with Kalman updating turned off (by setting it in the cfg file)

4. Analyze and plot state DA results
    1) Concatenate states for each ensemble member (for both DA and no-update ensembles):

        `$ cd ./tools/plot_analyze_results/`
        `$ python concat_updated_states.py <cfg_plot_EnKF> 1 <ens>`
      (OR  `$ python concat_updated_states_cellAvg.py <cfg_plot_EnKF> 1 <ens>` if only cellAvg states are saved from DA run)

    2) Concatenate history files (from each year to the whole period) and calculate daily runoff for each ensemble member (for both DA and no-update ensembles):

        `$ cd ./src`
        `$ python concat_EnKF_hist.py <cfg> <ens>`
        `$ python calculate_EnKF_hist_daily.py <cfg> <ens>`

    3) Calculate statistics for the openloop and perfect-state runs

        `./tools/plot_analyze_results/plot_synth_maps.ipynb with <cfg_plot_synth>`

    4) Pre-calculate CRPS statistics (for both DA and no-update ensembles):

        `$ cd ./tools/plot_analyze_results`
        `$ python calculate_crps.py <cfg_plot_EnKF> <var> <nproc>`

    5) Calculate and plot all statistics

        `$ ./tools/plot_analyze_results/plot_DA_maps_EnKF.ipynb with <cfg_plot_EnKF>`

5.  Run SMART rainfall correction

    `$ python hyak.prep_SMART_input.py <cfg>`
    `$ cd /SMART_output/prep_SMART`
    `$ matlab -nosplash -nodisplay -r run`
    `$ cd /civil/hydro/ymao/data_assim/src`
    `$ python postprocess_SMART.py $cfg <nproc>`

6.  Run post-processing
For each state and precipitation ensemble member, run:

    `$ python postprocess_EnKF.single_ens.py <run_da_cfg> <mpi_proc> <ens_prec> <ens_state>`

7. Analyze and plot post-processed ensemble runoff results
    1) Calculate daily runoff (from each year to the whole period) for each ensemble member:

        `$ cd ./src`
        `$ python calculate_post_hist_daily.py <cfg> <ens_prec> <ens_state>`

    2) Pre-calculate CRPS statistics (for post ensemble; the one for zero-update ensembles was calculated in step 4 already):

        `$ cd ./tools/plot_analyze_results`
        `$ python calculate_post_crps.py <cfg_plot_post> <var> <nproc>`

    6) Calculate and plot all statistics

        `$ ./tools/plot_analyze_results/plot_post_ens_maps.ipynb with <cfg_plot_EnKF>`


## Real-data:
1. Generate ensemble precipitation (for state EnKF)

    `$ python gen_ensemble_forcing.py <cfg> <ens>`

2. Run DA for state updating

    `$ python run_data_assim.py <cfg> <nproc> <mpi_proc> <debug> <restart> < save_cellAvg_state_only>`

If want to evaluate ensemble statistics, need to do a similar DA run with Kalman updating turned off (by setting it in the cfg file)

3. Analyze and plot EnKF results (before routing)

        `./tools/plot_analyze_results/plot_real_data_DA.ipynb with <cfg_EnKF_plot>`

4. Route streamflow
    1) Generate RVIC impulse response functions (only need to run this step once for each set of gauges and RVIC parameters):

        `$ rvic parameters <cfg_rvic_parameters>`

    2) Route steamflow (for both DA and no-update ensembles):

        `$ rvic convolution <cfg_rvic_convolution>`

5. Analyze and plot streamflow results from state updating

        `./tools/process_evaluation_data_ArkRed/plot_routed_flow_USGS_no_update.ipynb`
        `./tools/process_evaluation_data_ArkRed/plot_routed_flow_USGS.ipynb`



