function [corr_coef_negative] = API_with_true_rain_for_tuning(API_mean, bb, ist, ...
    rain_true, rain_observed, sm_observed)

%% Run API model with true rainfall for one pixel
% --- Some initialization --- %
API_true_rain(1:ist) = 0;  % [time]

% --- Run API model with true rainfall --- #
for k=2:ist
    API_COEFF = API_mean;  % right now only support consant API coefficient
    temp_API = API_short(API_true_rain(k-1), API_COEFF, bb, 1);
    API_true_rain(k) = temp_API + rain_true(k);
end

% --- Calculate correlation --- #
corr_coef = corrcoef(API_true_rain(~isnan(sm_observed)), ...
        sm_observed(~isnan(sm_observed)));
corr_coef_negative = - corr_coef(1, 2);
    
    