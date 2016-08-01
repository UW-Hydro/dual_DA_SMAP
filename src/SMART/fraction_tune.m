function fraction_fit = fraction_tune(fraction,sum_rain,sum_rain_sp,increment_sum)
rain_corrected = sum_rain_sp + (fraction*increment_sum); 
rain_corrected(rain_corrected <= 0) = 0;
%rain_corrected =
%rain_corrected.*(mean(sum_rain_sp)/mean(rain_corrected));%STANDARD DOES
%NOT INCLUDE THIS..KEEP COMMENTED OUT
fraction_fit=sqrt(mean((rain_corrected-sum_rain).^2));
%A = corrcoef(rain_corrected,sum_rain);
%fraction_fit = 1 - A(1,2)^2;
