function fraction_fit = fraction_tune(fraction,sum_rain,sum_rain_sp,increment_sum)
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
%   fraction_fit: RMS difference
% (Yixin)

rain_corrected = sum_rain_sp + (fraction*increment_sum); 
rain_corrected(rain_corrected <= 0) = 0;
%rain_corrected =
%rain_corrected.*(mean(sum_rain_sp)/mean(rain_corrected));%STANDARD DOES
%NOT INCLUDE THIS..KEEP COMMENTED OUT
fraction_fit=sqrt(mean((rain_corrected-sum_rain).^2));
%A = corrcoef(rain_corrected,sum_rain);
%fraction_fit = 1 - A(1,2)^2;
