function fraction_fit = fraction_tune_corrcoef(fraction,sum_rain,sum_rain_sp,increment_sum)
% This functions calculates the RMS difference between the independent
% rainfall data and the corrected observed rainfall data under a given
% lambda
% Inputs:
%   fraction: lambda
%   sum_rain: independent rainfall time series for each window (to tune against)
%   sum_rain_sp: observed rainfall time series for each window (to correct)
%   increment_sum: API increment time series for each window
%   NOTE: these three time series must have identical time indices (but need
% not to be continuous)
% Return:
%   fraction_fit: - corrcoef
% (Yixin)

rain_corrected = sum_rain_sp + (fraction*increment_sum); 
rain_corrected(rain_corrected <= 0) = 0;

% Fit correlation coefficient
fraction_fit = corrcoef(rain_corrected, sum_rain);
fraction_fit = - fraction_fit(1, 2);

