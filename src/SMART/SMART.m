
function [output_stat1, output_stat2] = SMART(varargin)

%% Process input arguments
p = inputParser;
% 'addParamValue' in old releases; 'addParameter' in new releases
% WARNING...some of these choice are not fully implemented
p.addParamValue('input_dataset', []);  % the input .mat file path; containing: 3 prec datasets; 2 soil moisture datasets; lidx; dnum
p.addParamValue('output_dataset', []);  % output .mat file path; containing corrected rainfall data
p.addParamValue('start_date', []);  % start date of simulation and data
p.addParamValue('end_date', []);  % end date of simulation and data
p.addParamValue('filter_flag', []);  % filter_flag 1)KF, 2)EnKF, 3)DI, 4)PART, 5)KF with RTS gap-filling, 6) EnKF with EnKS gap-filling, 7) PART - DIRECT RAINFALL
p.addParamValue('transform_flag', []);  % transform_flag 1) CDF, 2) seasonal 1&2, 3) bias 1&2, 4) seasonal CDF 
p.addParamValue('API_model_flag', []);  % API_model_flag 0) static 1) simple sine model, 2) Ta climatology, 3) PET climatologoy, 4) Ta-variation, 5) PET variation
p.addParamValue('lambda_flag', []);  % if = 999 then obtain lambda via fitting against "rain_indep", otherwise it sets a fixed value of lambda
p.addParamValue('NUMEN', []);  % NUMEN - number of ensembles used in EnKF or EnKS analysis...not used if filter_flag  = 1 or 3
p.addParamValue('Q_fixed', []);  % Q_fixed - if = 999 than whiten tune, otherwise it sets Q
p.addParamValue('P_inflation', []);
p.addParamValue('upper_bound_API', []);  % set to 99999 if do not want to set max soil moisture
p.addParamValue('logn_var', []);  % logn_var - variance of multiplicative ensemble perturbations...not sued if filter_flag = 1 or 3....setting to zero means all rainfall error is additive
p.addParamValue('slope_parameter_API', []);  % slope parameter API - not used if API_model_flag = 0
p.addParamValue('location_flag', []);  % location flag 0) CONUS, 1) AMMA, 2) Global 3) Australia 31 4) Australia 240 5) Australia, 0.25-degree continental
p.addParamValue('window_size', []);  % window size - default is one
p.addParamValue('API_mean', []);  % where API(t) = API_mean*API(t-1)^bb + rain(t)...default is 0.60
p.addParamValue('bb', []);  % where API(t) = API_mean*API(t-1)^bb + rain(t)...default is 0.60
p.addParamValue('API_range', []); % only used if API is varying seasonally
p.parse(varargin{:});
% Assign input arguments to variables
input_dataset = p.Results.input_dataset;
output_dataset = p.Results.output_dataset;
start_date = p.Results.start_date;
end_date = p.Results.end_date;
filter_flag = str2num(p.Results.filter_flag);
transform_flag = str2num(p.Results.transform_flag);
API_model_flag = str2num(p.Results.API_model_flag);
lambda_flag = str2num(p.Results.lambda_flag);
NUMEN = str2num(p.Results.NUMEN);
Q_fixed = str2num(p.Results.Q_fixed);
P_inflation = str2num(p.Results.P_inflation);
upper_bound_API = str2num(p.Results.upper_bound_API);
logn_var = str2num(p.Results.logn_var);
slope_parameter_API = str2num(p.Results.slope_parameter_API);
location_flag = str2num(p.Results.location_flag);
window_size = str2num(p.Results.window_size);
API_mean = str2num(p.Results.API_mean);
bb = str2num(p.Results.bb);
API_range = str2num(p.Results.API_range);

dump_flag = 0;

if (location_flag == 1) % Little Washita site
    load(input_dataset);
end

% Extract number of days
dnum1 = datenum(start_date);
dnum2 = datenum(end_date);
dnum = (dnum1:dnum2);
ist = numel(dnum);  % number of days Yixin

% Extract number of pixels
size_data = size(prec_orig);
numpixels = size_data(1);

for j=1:numpixels %space loop
    
    % REQUIRED INPUTS (TIME SERIES FOR EACH SPATIAL PIXEL)
    % (Missing input data is assumed to be nan)
    %1) rain_observed (satellite data to be corrected)
    %2) rain_indep (satellite data for calibration)
    %3) rain_true (ground gauge data for verification)
    %4) sma_observed (ascending soil moisture retrievals)
    %5) smd_observed (descending soil moisture retrievals)
    %6) sm_quality (standard error of soil moisture retrievals)
    
    %Note:
    %1) ascending and descending are combine into single time series below.
    %2) to start, ok to use "rain_true" as "rain_indep".
    %3) Also, ok to use default value of 0.04 m3m3 in "sm_quality"
    
    % Extract data for this pixel - Yixin
    rain_observed = prec_orig(j, :); % Satellite-based precipitation [1 * ntime];
    rain_indep = prec_for_tuning_lambda(j, :); % Calibration target
    rain_true = prec_true(j, :); % Used as benchmark
    sma_observed = sm_ascend(j, :); % Soil Moisture - Ascending
    smd_observed = sm_descend(j, :); % Soil Moisture - Descending
    sm_quality = sm_error(j, :);  % Soil moisture standard error
    
    % Change nan missing data to negative values
    % This is not ideal...should really change code to use nan for missing data
    rain_observed(isnan(rain_observed)) = -1;
    rain_indep(isnan(rain_indep)) = -1;
    rain_true(isnan(rain_true)) = -1;
    sma_observed(isnan(sma_observed)) = -1;
    smd_observed(isnan(smd_observed)) = -1;
    
    % The below data sources are used to defined to make the API coefficient vary in time
    % Only used for API_model_flag >= 2
    EVI_observed(1:ist) = -1;
    ta_observed(1:ist) = -1;
    PET_observed(1:ist) = -1;
    ta_observed_climatology(1:365) = -1;
    PET_observed_climatology(1:365) = -1;
    
%     water_fraction = 0;
%     % Spatial masking
%     if (water_fraction < 25 && sum(rain_indep(rain_indep > 0))...
%             > 20 && sum(rain_true(rain_true > 0)) > 20 ...
%             && sum(rain_observed(rain_observed > 0)) > 20 ...
%             && sqrt(var(sma_observed((sma_observed >=0))))...
%             > 0*1e-3 && sqrt(var(smd_observed((smd_observed >=0)))) > 0*1e-3) 
        
        rain_observed_hold = rain_observed;
        rain_observed_hold(rain_observed_hold < 0) = -99999; % set to -9999 to throw out windows with missing days (STANDARD)....set to 0 to keep them in the analysis with a default value of zero.
        
        rain_observed(rain_observed < 0) = 0; %for the rain time series feed into the KF...get rid of missing rain data...dataset must be complete (no missing values)
        rain_true(rain_true < 0) = -99999; % all calibration/evaluation is based on windows with no missing indep or true rainfall data
        rain_indep(rain_indep < 0) = -99999; % all calibration/evaluation is based on windows with no missing indep or true rainfall data
        rain_indep=rain_indep*mean(rain_observed_hold(rain_observed_hold >= 0))/mean(rain_indep(rain_indep >= 0)); %make sure RS precipitation products have same mean
        
        % Merge ascending and descending SM below to a daily product
        sm_observed(1:ist) = -1;
        for k=1:ist
            if (sma_observed(k) >= 0 && smd_observed(k) >= 0 );
                sm_observed(k) =  0.5*(smd_observed(k) + sma_observed(k));
            end;
            if (sma_observed(k)  < 0 && smd_observed(k) >= 0 );
                sm_observed(k) =  smd_observed(k); end;
            if (sma_observed(k) >= 0 && smd_observed(k) < 0 );
                sm_observed(k) =  sma_observed(k); end;
        end
        
        % Calculate Increments
        [increment_sum,increment_sum_hold,sum_rain,sum_rain_sp,sum_rain_sp_hold,sum_rain_sp2,increment_ens] = ...
            analysis(window_size,ist,filter_flag,transform_flag,API_model_flag,NUMEN,Q_fixed,P_inflation,...
            logn_var,bb,rain_observed,rain_observed_hold,rain_indep,rain_true,sm_observed,...
            ta_observed,ta_observed_climatology,PET_observed,PET_observed_climatology,EVI_observed,...
            API_mean,sm_quality, API_range, slope_parameter_API);
        
        % Correct Rainfall..Yixin, this sub-routine will have to be changed to accomodate ensemble
        [sum_rain_corrected,optimized_fraction] = ...
            correction(increment_sum,increment_sum_hold,sum_rain_sp,sum_rain_sp2,lambda_flag);
        
        RAIN_SMART_SMOS(:,j) = sum_rain_corrected(:);
%     end
end

save(output_dataset, 'RAIN_SMART_SMOS');  % [ntime, npixel]

