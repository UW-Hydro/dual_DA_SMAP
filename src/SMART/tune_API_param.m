function tune_API_param(varargin)

% This function tunes API_coeff to maximize the correlation coefficient 
% between the API time series (forced by true rainfall) and SM measurments
% If bb == -1, tune bb as well as API_COEFF; otherwise only tune API_COEFF

rng(22);

%% Process input arguments
p = inputParser;
% 'addParamValue' in old releases; 'addParameter' in new releases
% WARNING...some of these choice are not fully implemented
p.addParamValue('input_dataset', []);  % the input .mat file path; containing: 3 prec datasets; 2 soil moisture datasets; soil moisture error
p.addParamValue('output_dir', []);  % output directory; corrected rainfall data, innovation and lambda parameter will be written to this directory
p.addParamValue('start_time', []);  % start time of simulation and data; format: "YYYY-MM-DD HH:MM"
p.addParamValue('end_time', []);  % end time of simulation and data; format: "YYYY-MM-DD HH:MM"
p.addParamValue('time_step', []);  % Time step length in hour for all input data 
p.addParamValue('filter_flag', []);  % filter_flag 1)KF, 2)EnKF, 3)DI, 4)PART, 5)KF with RTS gap-filling, 6) EnKF with EnKS gap-filling, 7) PART - DIRECT RAINFALL
p.addParamValue('transform_flag', []);  % transform_flag 1) CDF, 2) seasonal 1&2, 3) bias 1&2, 4) seasonal CDF 
p.addParamValue('API_model_flag', []);  % API_model_flag 0) static 1) simple sine model, 2) Ta climatology, 3) PET climatologoy, 4) Ta-variation, 5) PET variation
p.addParamValue('lambda_flag', []);  % if = 999 then obtain lambda via fitting against "rain_indep", otherwise it sets a fixed value of lambda
p.addParameter('NUMEN', []);  % NUMEN - number of ensembles used in EnKF or EnKS analysis...not used if filter_flag  = 1 or 3
p.addParamValue('Q_fixed', []);  % Q_fixed - if = 999 than whiten tune, otherwise it sets Q
p.addParamValue('P_inflation', []);
p.addParamValue('upper_bound_API', []);  % set to 99999 if do not want to set max soil moisture
p.addParamValue('logn_var', []);  % logn_var - variance of multiplicative ensemble perturbations...not sued if filter_flag = 1 or 3....setting to zero means all rainfall error is additive
p.addParamValue('phi', []);  % precip perturbation autocorrelation
p.addParamValue('slope_parameter_API', []);  % slope parameter API - not used if API_model_flag = 0
p.addParamValue('location_flag', []);  % location flag 0) CONUS, 1) AMMA, 2) Global 3) Australia 31 4) Australia 240 5) Australia, 0.25-degree continental
p.addParamValue('window_size', []);  % window size - number of time steps in a window
p.addParamValue('API_mean', []);  % where API(t) = API_mean*API(t-1)^bb + rain(t)...default is 0.60
p.addParamValue('bb', []);  % where API(t) = API_mean*API(t-1)^bb + rain(t)...default is 0.60
p.addParamValue('API_range', []); % only used if API is varying seasonally
p.addParamValue('sep_sm_orbit', []); % 1 for separately rescale ascending and descending SM (this only makes sense for subdaily run); 0 for combining ascending and descending soil miosture products and SM obs appearing on the same timestep will be averaged
p.addParamValue('synth_meas_error', []); % synthetic meas. error standard deviation (in the API regime)
p.parse(varargin{:});
% Assign input arguments to variables
input_dataset = p.Results.input_dataset;
output_dir = p.Results.output_dir;
start_time = p.Results.start_time;
end_time = p.Results.end_time;
time_step = str2num(p.Results.time_step);
filter_flag = str2num(p.Results.filter_flag);
transform_flag = str2num(p.Results.transform_flag);
API_model_flag = str2num(p.Results.API_model_flag);
lambda_flag = str2num(p.Results.lambda_flag);
NUMEN = str2num(p.Results.NUMEN);
Q_fixed = str2num(p.Results.Q_fixed);
P_inflation = str2num(p.Results.P_inflation);
upper_bound_API = str2num(p.Results.upper_bound_API);
logn_var = str2num(p.Results.logn_var);
phi = str2num(p.Results.phi);
slope_parameter_API = str2num(p.Results.slope_parameter_API);
location_flag = str2num(p.Results.location_flag);
window_size = str2num(p.Results.window_size);
API_mean = str2num(p.Results.API_mean);
bb = str2num(p.Results.bb);
API_range = str2num(p.Results.API_range);
sep_sm_orbit = str2num(p.Results.sep_sm_orbit);
synth_meas_error = str2num(p.Results.synth_meas_error);

% Load input data
load(input_dataset);

% Extract number of days
dnum1 = datenum(start_time);
dnum2 = datenum(end_time);
dnum = (dnum1:(time_step/24):dnum2);
ist = numel(dnum);  % number of timesteps Yixin

% Extract number of pixels
size_data = size(prec_orig);
numpixels = size_data(1);

%% Some checks of input arguments
if (API_model_flag ~= 0)
    fprintf('Error: currently only support API_model_flag = 0');
    exit(1);
end

if (Q_fixed == 999)
    fprintf('Q_fixed must not be 999!');
    exit(1);
end

%% Initialize API_coeff
API_COEFF_tuned(1:numpixels) = 0;
if bb == -1
    bb_tuned(1:numpixels) = 1;
end

%% Tune API_coeff for each pixel individually
% --- Loop over each pixel --- %
for j=1:numpixels
    % ----- Rescale sm observations toward API with true rainfall ----- %
    % Extract data for this pixel
    rain_true = prec_true(j, 1:ist);
    rain_observed = prec_orig(j, 1:ist);
    sma_observed = sm_ascend(j, :);
    smd_observed = sm_descend(j, :);
    
    %total_mean_TA = mean(ta_observed_climatology(ta_observed_climatology > -100));%Global average high temperature
    %total_mean_PET = mean(PET_observed_climatology(PET_observed_climatology > -100));%Global average high temperature
    total_mean_TA = 288;%long term global average TA over land
    total_mean_PET = 270;
    lag = 0;
    ta_observed_climatology = 0;
    PET_observed_climatology = 0;
    EVI_observed(1:ist) = 0;
    R_DQX = sm_error(j, :);

    if (API_model_flag == 3 || API_model_flag == 5) %make sure you don't encounter -999 in PET
        if (numel(PET_observed_climatology(PET_observed_climatology == -999)) > 0)
            total_mean_PET = 0;
            PET_observed(1:ist) = 0;
            PET_observed_climatology(1:365) = 0;
        end
    end

    % If sep_sm_orbit = 0, merge ascending and descending SM below to
    % a single product, and then rescale
    if sep_sm_orbit == 0
        % Merge products
        sm_observed(1:ist) = nan;
        for k=1:ist
            if (~isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
                sm_observed(k) =  0.5*(smd_observed(k) + sma_observed(k));
            end;
            if (isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
                sm_observed(k) =  smd_observed(k); end;
            if (~isnan(sma_observed(k)) && isnan(smd_observed(k)) );
                sm_observed(k) =  sma_observed(k); end;
        end
        % Rescale
        [sm_observed_trans, R_API, API_COEFF, API_model] = rescale(sm_observed, time_step, ...
            transform_flag, API_model_flag, ist, rain_true, API_mean, API_range, lag, ...
            slope_parameter_API, ta_observed_climatology, ...
            PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
            EVI_observed, R_DQX);
    % If sep_sm_orbit = 1, rescale ascending & descending sm observations
    % separately; sm_observed only serves as an indicator for update timesteps
    else
        % Rescale ascending & descending separately
        [sma_observed_trans, R_API_a, API_COEFF_a, API_model] = rescale(sma_observed, time_step, ...
            transform_flag, API_model_flag, ist, rain_true, API_mean, API_range, lag, ...
            slope_parameter_API, ta_observed_climatology, ...
            PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
            EVI_observed, R_DQX);
        [smd_observed_trans, R_API_d, API_COEFF_d, API_model] = rescale(smd_observed, time_step, ...
            transform_flag, API_model_flag, ist, rain_true, API_mean, API_range, lag, ...
            slope_parameter_API, ta_observed_climatology, ...
            PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
            EVI_observed, R_DQX);
        % Put rescaled ascending & desceinding sm together
        sm_observed_trans(1:ist) = nan;
        R_API(1:ist) = 0;
        API_COEFF(1:ist) = API_mean;
        sm_observed(1:ist) = nan;
        for k=1:ist
            if (~isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
                fprintf('Error: When sep_sm_orbit, ascending and descending products cannot appear on the same timestep!');
                exit(1);
            end;
            if (isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
                sm_observed_trans(k) =  smd_observed_trans(k);
                R_API(k) = R_API_d(k);
                API_COEFF(k) = API_COEFF_d(k);
                sm_observed(k) = smd_observed(k);
            end;
            if (~isnan(sma_observed(k)) && isnan(smd_observed(k)) );
                sm_observed_trans(k) =  sma_observed_trans(k);
                R_API(k) = R_API_a(k);
                API_COEFF(k) = API_COEFF_a(k);
                sm_observed(k) = sma_observed(k);
            end;
        end
    end
       
    % --- Tune API parameters (toward unscaled SM meas) --- %
    if bb == -1
        A = [1 0; -1 0; 0 1; 0 -1];
        b = [0.9999; -0.01; 1.5; -0.5];
        x0 = [0.8, 1];
        x_tuned = fmincon(...
            @(x) API_with_true_rain_for_tuning(x(1), x(2), ist, ...
                rain_true, rain_observed, sm_observed), ...
            x0, A, b);
        API_COEFF_tuned(j) = x_tuned(1);
        bb_tuned(j) = x_tuned(2);
    else
        API_COEFF_tuned(j) = fminbnd(...
            @(x) API_with_true_rain_for_tuning(x, bb, ist, ...
                rain_true, rain_observed, sm_observed), 0.01, 0.9999, ...
            optimset('Display','off'));
    end
end

%% Save tuned API_COEFF_tuned
if bb == -1
    %API_param_tuned = [API_COEFF_tuned; bb_tuned];  % [2, npixel]
    save([output_dir '/API_param_tuned.mat'], 'API_COEFF_tuned', 'bb_tuned');  % [npixel]
else
    save([output_dir '/API_COEFF_tuned.mat'], 'API_COEFF_tuned');  % [npixel]
end



