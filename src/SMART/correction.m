function [sum_rain_corrected, sum_rain_corrected_ens, optimized_fraction] = correction(increment_sum,increment_sum_hold, increment_sum_ens, sum_rain_sp,sum_rain_indep,lambda_flag, filter_flag, NUMEN, rain_perturbed_sum_ens, lambda_tuning_target, correct_magnitude_only)

% Initialize corrected rainfall (sum in each window) (Yixin)
d = size(sum_rain_sp);
ist = d(2);
sum_rain_corrected(1:ist)=0;
sum_rain_corrected_ens(1:ist, 1:NUMEN) = 0;

% Find optimized lambda
% --- Only keep the time steps when there is an increment and there is
% rainfall in the independent rainfall data --- % Yixin
%%%%%%%%%%%%% Yixin HACK %%%%%%%%%%%%%%%
% Set -999 increment time steps to 0, so that the objective function calculated during tuning lambda is exactly the same as post-calculation after SMART
increment_sum_hold(increment_sum_hold < -500) = 0;
%%%%%%%%%%%%% HACK END %%%%%%%%%%%

% Whether correct rainfall only when orig_rain > 0
if (correct_magnitude_only == 1)
    increment_sum_hold(sum_rain_sp==0) = 0;
    increment_sum(sum_rain_sp==0) = 0;
    increment_sum_ens(sum_rain_sp==0, :) = 0;
end

increment_sum_hold_pr = increment_sum_hold(sum_rain_indep >= 0);
increment_sum_hold_subset = increment_sum_hold_pr(increment_sum_hold_pr > -500);

sum_rain_indep_pr = sum_rain_indep(sum_rain_indep >= 0);
sum_rain_indep_subset = sum_rain_indep_pr(increment_sum_hold_pr > -500);

sum_rain_sp_pr = sum_rain_sp(sum_rain_indep >= 0);
sum_rain_sp_subset = sum_rain_sp_pr(increment_sum_hold_pr > -500);

% --- find optimized lambda --- %
if (lambda_flag == 999)
    if (strcmp(lambda_tuning_target, 'rmse') == 1)
        optimized_fraction = fminbnd(@(x) fraction_tune(x,sum_rain_indep_subset,sum_rain_sp_subset,increment_sum_hold_subset),0.01,2.00,optimset('Display','off'));
    elseif (strcmp(lambda_tuning_target, 'corrcoef') == 1)
        optimized_fraction = fminbnd(@(x) fraction_tune_corrcoef(x,sum_rain_indep_subset,sum_rain_sp_subset,increment_sum_hold_subset),0.01,2.00,optimset('Display','off'));
    end
end
if (lambda_flag ~= 999); optimized_fraction = lambda_flag; end;

% Calculate corrected rainfall
increment_sum(increment_sum < -500) = 0;

for k=1:ist
    sum_rain_corrected(k) = sum_rain_sp(k)+(optimized_fraction*increment_sum(k));
    % If ensemble method, save ensemble of corrected rainfall
    % Add each ensemble member of increment to the corresponding perturbed
    % rainfall
    if (filter_flag == 2 || filter_flag == 6)
        sum_rain_corrected_ens(k, :) = rain_perturbed_sum_ens(k, :)+(optimized_fraction*increment_sum_ens(k, :));
    end
end

% Set negative rainfall to zero after correction (this introduces a bias)
sum_rain_corrected(sum_rain_corrected<0) = 0;
sum_rain_corrected_ens(sum_rain_corrected_ens<0) = 0;

end

% additive correction to remove bias
%hold2=sum(sum_rain_corrected);
%threshold = fminbnd(@(x) (hold2 - hold1 - (sum(sum_rain_corrected((find(sum_rain_corrected < x))))) - x*(length(find(sum_rain_corrected >= x))))^2,0,20);
%for k=(1:ist)
%    if (sum_rain_corrected(k) >= threshold)
%        sum_rain_corrected(k) = (sum_rain_corrected(k) - threshold);
%    else
%        sum_rain_corrected(k) = 0;
%    end
%end
