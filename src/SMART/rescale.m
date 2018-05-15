function [sm_observed_trans, R_API, API_COEFF] = rescale(sm_observed, time_step, ...
    transform_flag, API_model_flag, ist, rain_observed, API_mean, API_range, lag, ...
    slope_parameter_API, ta_observed_climatology, ...
    PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
    EVI_observed, R_DQX)

API_COEFF_HOLD(1:ist)=0;
API_COEFF(1:ist)=API_mean;
API_model(1:ist)=0;
R_API(1:ist)=0;
sm_observed_trans(1:ist)=0;

API_DOY(1:365)=0;
API2_DOY(1:365)=0;
mean_API(1:365)=0;
mean_sm_observed(1:365)=0;
var_sm_observed(1:365)=0;
sm_observed_DOY(1:365)=0;
sm2_observed_DOY(1:365)=0;
count_DOY_sm(1:365)=0;
count_DOY_API(1:365)=0;
var_API(1:365)=0;
DOY(1:365) = 0;

% sum for climatologies
DOY(1) = 365*((1+31)*0.002739726 - floor((1+31)*0.002739726));
for k=2:ist
    day = floor((k-1) * (time_step / 24)) + 1;  % calculate day index from timestep index (Yixin)
    DOY(k) = 365*((day+31)*0.002739726 - floor((day+31)*0.002739726));  % Yixin
    if (rain_observed(k) < 0);rain_observed(k) = 0;end
    
    % Calculate API coefficient gamma (Yixin)
    API_COEFF_HOLD(k) = API_mean + API_range*cos(2*pi*(DOY(k)-lag)/365);
    if (API_model_flag == 0); API_COEFF(k) = API_mean; end;
    if (API_model_flag == 1); API_COEFF(k) = API_mean + API_range*cos(2*pi*(DOY(k)-lag)/365); end;
    if (API_model_flag == 2 || API_model_flag == 4); API_COEFF(k) = API_mean - slope_parameter_API*(ta_observed_climatology(round(DOY(k))) - total_mean_TA); end;
    if (API_model_flag == 3 || API_model_flag == 5); API_COEFF(k) = API_mean - slope_parameter_API*(PET_observed_climatology(round(DOY(k))) - total_mean_PET); end;
    if (API_model_flag == 4)
        API_COEFF(k) = API_mean - 0.01*(ta_observed_climatology(round(DOY(k))) - total_mean_TA);%HARDWIRE
        API_COEFF(k) = API_COEFF(k) - slope_parameter_API*(ta_observed(k) - ta_observed_climatology(round(DOY(k))));
    end;
    if (API_model_flag == 5)
        API_COEFF(k) = API_mean - 0.01*(ta_observed_climatology(round(DOY(k))) - total_mean_TA);%HARDWIRE
        API_COEFF(k) = API_COEFF(k) - slope_parameter_API*(PET_observed(k) - PET_observed_climatology(round(DOY(k))));
    end;
    if (API_COEFF(k) > 1);API_COEFF(k) = 1;end;
    if (API_COEFF(k) < 0);API_COEFF(k) = 0;end;
    
    % Run API model for one time step with no Kalman filter update (Yixin)
    API_model(k) = sign(API_model(k-1))*(API_COEFF(k)-1)*abs(API_model(k-1))^bb + API_model(k-1) + rain_observed(k);
%     if (filter_flag == 2 || filter_flag == 4 || filter_flag ==6 ) % currently only for ensemble methods
%         Implicit_test = sign(API_model(k))*(API_COEFF(k)-1)*abs(API_model(k))^bb + API_model(k-1) + rain_observed(k);
%         if (abs(API_model(k) - Implicit_test) >= 5);
%             API_model(k) = API_short(API_model(k-1),API_COEFF(k),bb,24) + rain_observed(k);
%         end;
%     end;
    
    if (sm_observed(k) >= 0)
        sm_observed_DOY(round(DOY(k))) = sm_observed_DOY(round(DOY(k))) + sm_observed(k);
        sm2_observed_DOY(round(DOY(k))) = sm2_observed_DOY(round(DOY(k))) + sm_observed(k)*sm_observed(k);
        count_DOY_sm(round(DOY(k))) = count_DOY_sm(round(DOY(k))) + 1;
        % Calculate API_DOY values at SM-obs-available timesteps only
        % - Yixin
        API_DOY(round(DOY(k))) = API_DOY(round(DOY(k)))+API_model(k);
        API2_DOY(round(DOY(k))) = API2_DOY(round(DOY(k)))+API_model(k)*API_model(k);
        count_DOY_API(round(DOY(k))) = count_DOY_API(round(DOY(k))) + 1;
    end
end

% Calculate all moment statistics using API)model values at
% SM-obs-available timesteps only - Yixin
total_sd_ratio = (sqrt(var(API_model(sm_observed >= 0)))/sqrt(var(sm_observed((sm_observed >= 0)))));
total_mean_sm_observed = mean(sm_observed((sm_observed >= 0)));
total_var_sm_observed = var(sm_observed((sm_observed >= 0)));
total_mean_API = mean(API_model(sm_observed >= 0));

% average climatologies within 31-day moving windows
for k=1:365
    count_in_window_sm_observed=0;
    count_in_window_API=0;
    for i=k-15:k+15
        i_prime=i;
        if (i<1); i_prime=i+365; end;
        if (i>365); i_prime=i-365; end;
        if (count_DOY_sm(i_prime) > 0)
            count_in_window_sm_observed = count_in_window_sm_observed + count_DOY_sm(i_prime);
            mean_sm_observed(k) = mean_sm_observed(k) + sm_observed_DOY(i_prime);
            var_sm_observed(k) = var_sm_observed(k) + sm2_observed_DOY(i_prime);
        end
        count_in_window_API = count_in_window_API + count_DOY_API(i_prime);
        mean_API(k) = mean_API(k) + API_DOY(i_prime);
        var_API(k) = var_API(k) + API2_DOY(i_prime);
    end
    if (count_in_window_sm_observed > 0)
        mean_sm_observed(k)=mean_sm_observed(k)/count_in_window_sm_observed;
        var_sm_observed(k)=(count_in_window_sm_observed/(count_in_window_sm_observed-1))*(var_sm_observed(k)/count_in_window_sm_observed - mean_sm_observed(k)*mean_sm_observed(k));
    else
        mean_sm_observed(k) = total_mean_sm_observed;
        var_sm_observed(k) = total_var_sm_observed;
    end
    mean_API(k)=mean_API(k)/count_in_window_API;
    var_API(k)=(count_in_window_API/(count_in_window_API-1))*(var_API(k)/count_in_window_API - mean_API(k)*mean_API(k));
end

% rescaling observations and defining R
if (transform_flag == 1)
    RS_sort=sort(sm_observed((sm_observed >= 0)));
    API_sort=sort(transpose(API_model((sm_observed >= 0))));
end

for k=2:ist
    
    if (EVI_observed(1+floor(k/30.5)) > 0.00) % every month pull a new EVI
        sqrt_R = 0.20*EVI_observed(1 + floor(k/30.5));
    else
        R = max(R_DQX(k)^2, 0.00); %Use quality control information to define R
    end
    R_API(k) = R*(total_sd_ratio)^2;
    
    sm_observed_trans(k) = -1;
    if (transform_flag == 1 && sm_observed(k) >= 0); sm_observed_trans(k) = mean(API_sort((RS_sort == sm_observed(k))));end;
    if (transform_flag == 2 && sm_observed(k) >= 0); sm_observed_trans(k) = (sm_observed(k) - mean_sm_observed(round(DOY(k)))) * total_sd_ratio + mean_API(round(DOY(k))); end;
    if (transform_flag == 3 && sm_observed(k) >= 0); sm_observed_trans(k) = (sm_observed(k) - total_mean_sm_observed) * total_sd_ratio + total_mean_API; end;
    
    if (transform_flag == 4 && sm_observed(k) >= 0)
        % Yixin: here is doing 91-day-window CDF matching
        % Get all sm_obs and API data within multi-year 91-day window 
        delta_DOY = abs(DOY - DOY(k));
        delta_DOY(delta_DOY > 182.5) = 365 - delta_DOY(delta_DOY > 182.5);
        sm_observed_subset = sm_observed(abs(delta_DOY) <= 45);
        API_model_subset = API_model(abs(delta_DOY) <= 45);
        % Sort sm_obs and API (at sm_obs-available steps)
        RS_sort=sort(sm_observed_subset((sm_observed_subset >= 0)));
        API_sort=sort(transpose(API_model_subset((sm_observed_subset) >= 0)));
        % Rescale
        sm_observed_trans(k) = mean(API_sort((RS_sort == sm_observed(k))));
    end
    
    if (transform_flag == 5)
        if (var_API(round(DOY(k))) > 0 && var_sm_observed(round(DOY(k))) > 0)
            if (sm_observed(k) >= 0); sm_observed_trans(k) = (sm_observed(k) - mean_sm_observed(round(DOY(k)))) * sqrt(var_API(round(DOY(k))))/sqrt(var_sm_observed(round(DOY(k)))) + mean_API(round(DOY(k))); end;
        else
            if (sm_observed(k) >= 0); sm_observed_trans(k) = (sm_observed(k) - mean_sm_observed(round(DOY(k)))) * total_sd_ratio + mean_API(round(DOY(k))); end;
        end
    end
end






