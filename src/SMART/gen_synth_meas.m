function gen_synth_meas(varargin)

% This function generates synthetic measurement based on API - this is for
% identical-twin experiments for SMART
% NOTE: only "prec_orig" in the input data will be used here; other input
% data to SMART will be regenerated
%
% The output from this function is a saved .mat data containing
% SMART-format input data, where "prec_for_tuning_lambda" = prec_true, all
% synthetic measurements will be saved to "sm_ascend", and "sm_error" is
% spatially and temporally constant value in the API regime (input to this
% function)

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
p.addParamValue('NUMEN', []);  % NUMEN - number of ensembles used in EnKF or EnKS analysis...not used if filter_flag  = 1 or 3
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

%% Generate synthetic truth - run API model with forcing and state perturbation
% --- Some initialization --- %
prec_true = prec_orig;  % [pixel, time]
API_true(1:numpixels, 1:ist) = 0;  % [pixel, time]
synth_meas(1:numpixels, 1:ist) = nan;  % [pixel, time]

% --- Loop over each pixel --- %
for j=1:numpixels
    % Extract data for this pixel
    rain_observed = prec_orig(j, 1:ist);
    sm_quality = sm_error(j, 1:ist);
    
    % Perturb rainfall for all timesteps
    mult_factor = generate_prec_lognormal_multiplier(logn_var, phi, ist);
    prec_true(j, :) = mult_factor .* rain_observed;
    
    % Run API model (with percipitation and state perturbation)
    for k=2:ist
        API_COEFF = API_mean;  % right now only support consant API coefficient
        temp_API = API_short(API_true(j, k-1), API_COEFF, bb, 1);
        API_true(j, k) = temp_API + prec_true(j, k) + ...
            sqrt(Q_fixed)*randn(1);  % No P_inflation
    end
    
    % Generate synthetic measurements, daily
    for k = 1:ist
        if mod(k*time_step, 24) == 0
            synth_meas(j, k) = API_true(j, k) + ...
                synth_meas_error * randn(1);
        end
    end
end

%% Save SMART-input-format data file, plus true states
prec_for_tuning_lambda = prec_true;
sm_ascend = synth_meas;
sm_descend(1:numpixels, 1:ist) = nan;
sm_error(1:numpixels, 1:ist) = synth_meas_error;

save([output_dir '/smart_input_synth_from_API.mat'], ...
    'prec_orig', 'prec_true', 'prec_for_tuning_lambda', 'sm_ascend', ...
    'sm_descend', 'sm_error', 'API_true');
