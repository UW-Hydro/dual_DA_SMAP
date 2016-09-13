function SMART(varargin)

%% Process input arguments
p = inputParser;
% 'addParamValue' in old releases; 'addParameter' in new releases
% WARNING...some of these choice are not fully implemented
p.addParamValue('input_dataset', []);  % the input .mat file path; containing: 3 prec datasets; 2 soil moisture datasets;
p.addParamValue('output_dataset', []);  % output .mat file path; containing corrected rainfall data
p.addParamValue('start_date', []);  % start date of simulation and data
p.addParamValue('end_date', []);  % end date of simulation and data
p.addParamValue('filter_flag', []);  % filter_flag 1)KF, 2)EnKF, 3)DI, 4)PART, 5)RTS, 6) EnKS, 7) PART - DIRECT RAINFALL
p.addParamValue('transform_flag', []);  % transform_flag 1) CDF, 2) seasonal 1&2, 3) bias 1&2, 4) seasonal CDF 
p.addParamValue('API_model_flag', []);  % API_model_flag 0) static 1) simple sine model, 2) Ta climatology, 3) PET climatologoy, 4) Ta-variation, 5) PET variation
p.addParamValue('lambda_flag', []);  % lambda_flag 0) optimized, 1) PERSIANN, 2) 0.60
p.addParamValue('NUMEN', []);  % NUMEN - number of ensembles used in EnKF or EnKS analysis...not used if filter_flag  = 1 or 3
p.addParamValue('Q_fixed', []);  % Q_fixed - if = 999 than whiten tune, otherwise it sets Q
p.addParamValue('P_inflation', []);
p.addParamValue('upper_bound_API', []);  % set to 99999 if do not want to set max soil moisture
p.addParamValue('logn_var', []);  % logn_var - variance of multiplicative ensemble perturbations...not sued if filter_flag = 1 or 3....setting to zero means all rainfall error is additive
p.addParamValue('slope_parameter_API', []);  % slope parameter API - not used if API_model_flag = 0
p.addParamValue('location_flag', []);  % location flag 0) CONUS, 1) AMMA, 2) Global 3) Australia 31 4) Australia 240 5) Australia, 0.25-degree continental
p.addParamValue('window_size', []);  % window size - time scale at which rainfall correction is applied 3 to 10 days is recommended
p.addParamValue('API_mean', []);
p.addParamValue('API_range', []);
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
API_range = str2num(p.Results.API_range);

% Turn on/off diagnostic data dump
dump_flag = 1; % 0;

%% Australia - 0.25 Degree Continental Scale
if (location_flag == 5)
    load(input_dataset);
    % Yixin dnum1 = dnum; % Save date number of input in a new dnum1
    % Yixin: get number of pixels from data matrix instead
    % numpixels = numel(lidx); % Number of land pixels of 0.25-deg Australia = 11125
    
    dnum1 = datenum(start_date);
    dnum2 = datenum(end_date);
    dnum = (dnum1:dnum2);
    nd = numel(dnum);  % number of days Yixin
    ist = nd;
    window_ist = floor(ist/window_size);
    
    RAIN_1 = prec_orig'; % Satellite-based precipitation [ntime * npixel]
    RAIN_2 = prec_for_tuning_lambda'; % Calibration target
    RAIN_3 = prec_true'; % Used as benchmark
    SMOBS_A = sm_ascend'; % Soil Moisture - Ascending
    SMOBS_D = sm_descend'; % Soil Moisture - Descending
   
    clear prec_orig prec_for_tuning_lambda prec_true sm_ascend sm_descend;

    % Get number of pixels
    size_data = size(RAIN_1);
    numpixels = size_data(2);
    
    RAIN_1(find(isnan(RAIN_1))) = -999;
    RAIN_2(find(isnan(RAIN_2))) = -999;
    RAIN_3(find(isnan(RAIN_3))) = -999;   
    SMOBS_A(find(isnan(SMOBS_A))) = -999;
    SMOBS_D(find(isnan(SMOBS_D))) = -999;
end

%% 1. SMOS RE02 ASC-DSC composite

% Initialize corrected rainfall
RAIN_SMART_SMOS = zeros(window_ist,1);

for j=1:numpixels % space loop

    % Extract data for this pixel
    rain_observed = RAIN_1(:,j); % Satellite rainfall here
    rain_indep = RAIN_2(:,j); % Independent rainfall here/calibration target
    rain_true = RAIN_3(:,j); % RAIN_2(:,j); % AWAP rainfall
    sma_observed = SMOBS_A(:,j);
    smd_observed = SMOBS_D(:,j);

    EVI_observed(1:ist) = -1; %filler - not using EVI currently
    ta_observed(1:ist) = -1; %filler - not using currently
    PET_observed(1:ist) = -1; %filler - not using currently
    ta_observed_climatology(1:365) = -1; %filler - not using currently
    PET_observed_climatology(1:365) = -1; %filler - not using currently

    rain_observed(isnan(rain_observed)) = -1;
    rain_indep(isnan(rain_indep)) = -1;
    rain_true(isnan(rain_true)) = -1;
    sma_observed(isnan(sma_observed)) = -1;
    smd_observed(isnan(smd_observed)) = -1;
 
    % Not currently using water fraction
    water_fraction = 0;
 
    % Spatial mask to eliminate pixels lacking data
%    if (water_fraction < 25 && sum(rain_indep(rain_indep > 0))...
%            > 20 && sum(rain_true(rain_true > 0)) > 20 ...
%            && sum(rain_observed(rain_observed > 0)) > 20 ...
%            && sqrt(var(sma_observed((sma_observed >=0))))...
%            > 0*1e-3 && sqrt(var(smd_observed((smd_observed >=0)))) > 0*1e-3) %&& sqrt(var(ERSa_observed((ERSa_observed >=0)))) > 0*1e-3 && sqrt(var(ERSd_observed((ERSd_observed >=0)))) > 0*1e-3)

        rain_observed_hold = rain_observed;
        rain_observed_hold(rain_observed_hold < 0) = -99999; % set to -9999 to throw out windows with missing days (STANDARD)....set to 0 to keep them in the analysis with a default value of zero.

        rain_observed(rain_observed < 0) = 0; %for the time series feed into the KF...get rid of missing rain data...dataset must be complete (no missing values)
        rain_true(rain_true < 0) = -99999; % all calibration/evaluation is based on windows with no missing indep or true rainfall data
        rain_indep(rain_indep < 0) = -99999; % all calibration/evaluation is based on windows with no missing indep or true rainfall data 
        rain_indep=rain_indep*mean(rain_observed_hold(rain_observed_hold >= 0))/mean(rain_indep(rain_indep >= 0)); %make sure RS precipitation products have same mean
                
        sm_observed(1:ist) = -1;
        ERS_observed(1:ist) = -1;
        
        % Naive merging ascending and descending SM below to a daily product
        % If one of ascending and descending data is available at a certain
        % day, take it; if both are available on the same day, take the
        % mean (Yixin)
        for k=1:ist
            if (sma_observed(k) >= 0 && smd_observed(k) >= 0); 
                sm_observed(k) =  0.5*(smd_observed(k) + sma_observed(k)); 
            end;
            if (sma_observed(k)  < 0 && smd_observed(k) >= 0); 
                sm_observed(k) =  smd_observed(k); end;
            if (sma_observed(k) >= 0 && smd_observed(k) < 0);  
                sm_observed(k) =  sma_observed(k); end;
        end        

        % Run the filter to calculate increments....time loop is in this
        % function
        [increment_sum,increment_sum_hold,sum_rain,sum_rain_sp,sum_rain_sp_hold,sum_rain_sp2] = ...
            analysis(API_mean, API_range, window_size,ist,filter_flag,transform_flag,API_model_flag,NUMEN,Q_fixed,P_inflation,...
            logn_var,upper_bound_API,rain_observed,rain_observed_hold,rain_indep,rain_true,sm_observed,ERS_observed,...
            ta_observed,ta_observed_climatology,PET_observed,PET_observed_climatology,EVI_observed,slope_parameter_API,location_flag);
        
        % Apply increments to correct rainfall...time loop is in this
        % function
        [sum_rain_corrected,optimized_fraction] = ...
            correction(increment_sum,increment_sum_hold,sum_rain_sp,sum_rain_sp2,lambda_flag);
     
        % This is the corrected rainfall...below we evaluate it via
        % comparisons to an independent rainfall product
        RAIN_SMART_SMOS(:,j) = sum_rain_corrected(:);
%    else
%        RAIN_SMART_SMOS(:,j) = -1;
%    end
end

save(output_dataset, 'RAIN_SMART_SMOS');


