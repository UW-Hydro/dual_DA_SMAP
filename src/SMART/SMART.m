
function SMART(varargin)

rng(11);

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
p.addParamValue('slope_parameter_API', []);  % slope parameter API - not used if API_model_flag = 0
p.addParamValue('location_flag', []);  % location flag 0) CONUS, 1) AMMA, 2) Global 3) Australia 31 4) Australia 240 5) Australia, 0.25-degree continental
p.addParamValue('window_size', []);  % window size - number of time steps in a window
p.addParamValue('API_mean', []);  % where API(t) = API_mean*API(t-1)^bb + rain(t)...default is 0.60
p.addParamValue('bb', []);  % where API(t) = API_mean*API(t-1)^bb + rain(t)...default is 0.60
p.addParamValue('API_range', []); % only used if API is varying seasonally
p.addParamValue('sep_sm_orbit', []); % 1 for separately rescale ascending and descending SM (this only makes sense for subdaily run); 0 for combining ascending and descending soil miosture products and SM obs appearing on the same timestep will be averaged
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
slope_parameter_API = str2num(p.Results.slope_parameter_API);
location_flag = str2num(p.Results.location_flag);
window_size = str2num(p.Results.window_size);
API_mean = str2num(p.Results.API_mean);
bb = str2num(p.Results.bb);
API_range = str2num(p.Results.API_range);
sep_sm_orbit = str2num(p.Results.sep_sm_orbit);

dump_flag = 0;

if (location_flag == 1) % Little Washita site
    load(input_dataset);
end

% Extract number of days
dnum1 = datenum(start_time);
dnum2 = datenum(end_time);
dnum = (dnum1:(time_step/24):dnum2);
ist = numel(dnum);  % number of timesteps Yixin

% Extract number of pixels
size_data = size(prec_orig);
numpixels = size_data(1);

% Initialize some matrices
innovation(1:ist, 1:numpixels) = 0;
lambda(1:numpixels) = 0;

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
    rain_observed = prec_orig(j, 1:ist); % Satellite-based precipitation [1 * ntime];
    rain_indep = prec_for_tuning_lambda(j, 1:ist); % Calibration target
    rain_true = prec_true(j, 1:ist); % Used as benchmark
    sma_observed = sm_ascend(j, 1:ist); % Soil Moisture - Ascending
    smd_observed = sm_descend(j, 1:ist); % Soil Moisture - Descending
    sm_quality = sm_error(j, 1:ist);  % Soil moisture standard error
    
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
        
        % Calculate Increments
        [increment_sum,increment_sum_hold,sum_rain,sum_rain_sp,sum_rain_sp_hold,sum_rain_sp2,increment_sum_ens, innovation(:, j), innovation_not_norm, rain_perturbed_sum_ens] = ...
            analysis(window_size,ist,filter_flag,transform_flag,API_model_flag,NUMEN,Q_fixed,P_inflation,...
            logn_var,bb,rain_observed,rain_observed_hold,rain_indep,rain_true,sep_sm_orbit,sma_observed,smd_observed,...
            ta_observed,ta_observed_climatology,PET_observed,PET_observed_climatology,EVI_observed,...
            API_mean,sm_quality, API_range, slope_parameter_API, time_step);
        
        % Correct Rainfall
        [sum_rain_corrected, sum_rain_corrected_ens, optimized_fraction] = ...
            correction(increment_sum,increment_sum_hold,increment_sum_ens, sum_rain_sp,sum_rain_sp2,lambda_flag, filter_flag, NUMEN, rain_perturbed_sum_ens);
        lambda(j) = optimized_fraction;
        
        % Rescale corrected rainfall
        mask_sum_rain = sum_rain >= 0; %need benchmark to be there
        mask_sum_rain_sp = sum_rain_sp_hold >= 0; %need sat precip to be there (at least one)...
        mask_time_total_val = mask_sum_rain_sp.*mask_sum_rain; %create final mask for validation (Yixin: this is the timesteps when both true and observed rainfall >= 0)
        
        % For validation
        sum_rain_subset = sum_rain(mask_time_total_val > 0 );
        sum_rain_sp_subset = sum_rain_sp(mask_time_total_val > 0); 
        sum_rain_corrected_subset = sum_rain_corrected(mask_time_total_val > 0); 
                
        % This rescales corrected rainfall to have same mean as uncorrected (satellite rainfall) 
        % Due to truncation effects (at zero)...SMART tends to overestimate
        % rainfall...this corrects for that spurious effect
        hold1=mean(sum_rain_corrected_subset);
        hold2=mean(sum_rain_sp_subset);
        %hold3=mean(sum_rain_subset);
        sum_rain_corrected_subset=sum_rain_corrected_subset.*(hold2/hold1);
        % Put rescaled corrected rainfall back to matrix (Yixin)
        sum_rain_corrected(mask_time_total_val > 0) = sum_rain_corrected_subset;
        %sum_rain_corrected_subset_cal=sum_rain_corrected_subset_cal.*(hold2/hold1); 
        % Rescale for each corrected rainfall ensemble member (Yixin)
        for e=1:NUMEN
            sum_rain_corrected_ens_subset = sum_rain_corrected_ens(mask_time_total_val > 0, e);
            hold_ens = mean(sum_rain_corrected_ens_subset);
            sum_rain_corrected_ens_subset = sum_rain_corrected_ens_subset .*(hold2 / hold_ens);
            % Put rescaled rainfall back to matrix
            sum_rain_corrected_ens(mask_time_total_val > 0, e) = sum_rain_corrected_ens_subset;
        end
        
        RAIN_SMART_SMOS(:,j) = sum_rain_corrected(:);
        RAIN_SMART_SMOS_ENS(:, :, j) = sum_rain_corrected_ens(:, :);
        INCREMENT_SUM(:, j) = increment_sum(:);
        
        % Calculate calibration metric...use only when trying to calibrate API parameters again "rain_indep"
        % mask_sum_rain_sp2 = sum_rain_sp2 >=0; %need "independent" sat precip to be there (at least one)...for calibration
        % mask_time_total_cal = mask_sum_rain_sp2; %create final mask for validation
        % sum_rain_subset_cal = sum_rain(mask_time_total_cal > 0 ); %apply final mask
        % sum_rain_corrected_subset_cal = sum_rain_corrected(mask_time_total_cal > 0); %apply final mask
        % output_cal=sqrt(mean((sum_rain_corrected_subset_cal-sum_rain_subset_cal).^2));
        
        %TO DUMP ALL QUALIFYING windows into output file
        if (dump_flag  == 1)
            for k=1:numel(sum_rain_subset)
                fprintf(output,'%f %f %f \n',sum_rain_subset(k),sum_rain_sp_subset(k),sum_rain_corrected_subset(k));
            end
        end
        
%         % Everything from here to bottom is simply the calculation of validation metrics...this should probably be put into a dedicated sub-routine
%         
%         % Calculate validation metrics
%         A=corrcoef(sum_rain_subset,sum_rain_sp_subset);
%         if numel(A)==1
%             A=[A,0;0,A];
%         end
%         B=corrcoef(sum_rain_subset,sum_rain_corrected_subset);
%         if numel(B)==1
%             B=[B,0;0,B];
%         end
%         C=sqrt(mean((sum_rain_subset-sum_rain_sp_subset).^2));
%         D=sqrt(mean((sum_rain_subset-sum_rain_corrected_subset).^2));
%         E=mean(sum_rain_subset);
%         F=mean(sum_rain_sp_subset);
%         G=mean(sum_rain_corrected_subset);
%         H=var(sum_rain_subset);
%         I=var(sum_rain_sp_subset);
%         J=var(sum_rain_corrected_subset);
%         
%         % Calculate POD, FAR and CSI for various accumulation thresholds
%         
%         % Define rain event thresholds....
%         %sum_rain_nonzero_sort = sort(sum_rain_sp(sum_rain_sp > 2));  % USED THIS IN OLD RUNS
%         sum_rain_nonzero_sort = sort(sum_rain_subset(sum_rain_subset > 2)); %THIS IS STANDARD
%         number_total = numel(sum_rain_nonzero_sort);
%         
%         FAR_corr(1:12)=0;
%         POD_corr(1:12)=0;
%         CSI_corr(1:12)=0;
%         HR_corr(1:12)=0;
%         E_CSI_corr(1:12)=0;
%         FAR(1:12)=0;
%         POD(1:12)=0;
%         CSI(1:12)=0;
%         HR(1:12)=0;
%         E_CSI(1:12)=0;
%         event_threshold(1:12)=0;
%         
%         percentile(1)=0.00;
%         percentile(2)=0.10;
%         percentile(3)=0.20;
%         percentile(4)=0.30;
%         percentile(5)=0.40;
%         percentile(6)=0.50;
%         percentile(7)=0.60;
%         percentile(8)=0.70;
%         percentile(9)=0.80;
%         percentile(10)=0.90;
%         percentile(11)=0.95;
%         percentile(12)=0.99;
%         
%         if (numel(sum_rain_nonzero_sort) > 0)
%             for k=1:12
%                 event_threshold(k) =  sum_rain_nonzero_sort(floor(number_total*percentile(k))+1);
%                 floor(number_total*percentile(k))+1;
%                 number_hit_corr =   numel(find(sum_rain_corrected_subset((sum_rain_subset >= event_threshold(k))) >= event_threshold(k)));
%                 number_false_corr = numel(find(sum_rain_corrected_subset((sum_rain_subset < event_threshold(k)))  >= event_threshold(k)));
%                 number_miss_corr =  numel(find(sum_rain_corrected_subset((sum_rain_subset >= event_threshold(k)))  < event_threshold(k)));
%                 number_hit_no_corr = numel(find(sum_rain_corrected_subset((sum_rain_subset < event_threshold(k)))  < event_threshold(k)));
%                 
%                 FAR_corr(k) =       number_false_corr/(number_hit_corr + number_false_corr);
%                 if (isnan(FAR_corr(k))); FAR_corr(k) = 0; end; %STANDARD
%                 POD_corr(k) =       number_hit_corr/(number_miss_corr + number_hit_corr);
%                 CSI_corr(k) =       number_hit_corr/(number_miss_corr + number_false_corr + number_hit_corr);
%                 HR_corr(k) =        (number_hit_corr + number_hit_no_corr)/(numel(sum_rain_subset));
%                 A_R_corr =          (number_hit_corr + number_false_corr)*(number_hit_corr+number_miss_corr) / numel(sum_rain_subset);
%                 E_CSI_corr(k) =     (number_hit_corr - A_R_corr)/(number_miss_corr + number_false_corr + number_hit_corr - A_R_corr);
%                 
%                 number_hit =        numel(find(sum_rain_sp_subset((sum_rain_subset >= event_threshold(k))) >= event_threshold(k)));
%                 number_false =      numel(find(sum_rain_sp_subset((sum_rain_subset < event_threshold(k))) >= event_threshold(k)));
%                 number_miss =       numel(find(sum_rain_sp_subset((sum_rain_subset >= event_threshold(k))) < event_threshold(k)));
%                 number_hit_no =     numel(find(sum_rain_sp_subset((sum_rain_subset < event_threshold(k))) < event_threshold(k)));
%                 
%                 FAR(k) =            number_false/(number_hit + number_false);
%                 if (isnan(FAR(k))); FAR(k) = 0; end; %STANDARD
%                 POD(k) =            number_hit/(number_miss + number_hit);
%                 CSI(k) =            number_hit/(number_miss + number_false + number_hit);
%                 HR(k) =             (number_hit + number_hit_no)/(numel(sum_rain_subset));
%                 A_R =               (number_hit + number_false)*(number_hit+number_miss) / numel(sum_rain_subset);
%                 E_CSI(k) =          (number_hit - A_R)/(number_miss + number_false + number_hit - A_R);
%             end
%         else
%             for k=1:12
%                 FAR(k) = 0;
%                 POD(k) = 0;
%                 CSI(k) = 0;
%                 FAR_corr(k) = 0;
%                 POD_corr(k) = 0;
%                 CSI_corr(k) = 0;
%             end
%         end
%         
%         % Output metrics
%         fprintf(1,'%f %f %f  \n',-A(1,2)+B(1,2),-C+D,optimized_fraction)
%     end       
end

%% 
save([output_dir '/SMART_corrected_rainfall.mat'], 'RAIN_SMART_SMOS');  % [ntime, npixel]
save([output_dir '/SMART_corrected_rainfall_ens.mat'], 'RAIN_SMART_SMOS_ENS');  % [ntime, n_ens, npixel]
save([output_dir '/innovation.mat'], 'innovation');  % [ntime, npixel]
save([output_dir '/lambda.mat'], 'lambda');  % [npixel]
save([output_dir '/increment_sum.mat'], 'INCREMENT_SUM');  % [ntime, npixel]  

