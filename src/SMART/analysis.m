function [increment_sum,increment_sum_hold,sum_rain,sum_rain_sp,sum_rain_sp_hold,sum_rain_indep] =...
    analysis(API_mean, API_range, window_size,ist,filter_flag,transform_flag,API_model_flag,NUMEN,Q_fixed,P_inflation_fixed,...
    logn_var_constant,upper_bound_API,rain_observed,rain_observed_hold,rain_indep,rain_true,sm_observed,...
    ERS_observed,ta_observed,ta_observed_climatology,PET_observed,PET_observed_climatology,EVI_observed,...
    slope_parameter_API,location_flag)

% Very important hardwire!
% Should really move this
%API_mean = 0.60; % HARDWIRE FOR AUSTRALIA
%API_range = 0.15;

lag = 0.00; %lag=zero gives the correct seasonality
API_estimate_flag = 0;

%if (filter_flag == 2  || filter_flag == 4 || filter_flag == 6)
if (filter_flag > 1)
    API_filter_EnKF(1:ist,1:NUMEN)=0;
    API_filter_PART(1:ist,1:NUMEN)=0;
    API_COEFF_E(1:ist,1:NUMEN)=0.00;
    API_COEFF_V(1:ist,1:NUMEN)=0.80;
    ones(1:NUMEN)=1;
end

API_filter_f(1:ist)=0;
RTS_API(1:ist)=0;
R_API(1:ist)=0;
API_model(1:ist)=0;
API_COEFF(1:ist)=API_mean;
API_filter(1:ist)=0;
sm_observed_trans(1:ist)=0;
ERS_observed_trans(1:ist)=0;
increment(1:ist)=0;
increment_EnKS(1:ist)=0;
increment_hold(1:ist)=0;
innovation1(1:ist)=0;
API_COEFF_HOLD(1:ist)=0;

P(1:ist)=0;
Pf(1:ist)=0;
K(1:ist,1:2)=0;
K2(1:ist,1:2)=0;
CYM_API_COEFF(1:ist) = 0;
K_API_COEFF(1:ist) = 0;
sum_rain(1:floor(ist/window_size)) = 0;
sum_rain_sp(1:floor(ist/window_size)) = 0;
sum_rain_sp_hold(1:floor(ist/window_size)) = 0;
sum_rain_indep(1:floor(ist/window_size)) = 0;
increment_sum(1:floor(ist/window_size)) = 0;
API_DOY(1:365)=0;
API2_DOY(1:365)=0;
mean_API(1:365)=0;
API2_DOY(1:365)=0;
mean_sm_observed(1:365)=0;
mean_ERS_observed(1:365)=0;
ERS2_observed_DOY(1:365)=0;
ERS_observed_DOY(1:365)=0;
sm2_observed_DOY(1:365)=0;
var_sm_observed(1:365)=0;
var_ERS_observed(1:365)=0;
sm_observed_DOY(1:365)=0;
sm2_observed_DOY(1:365)=0;
count_DOY_sm(1:365)=0;
count_DOY_API(1:365)=0;
count_DOY_ERS(1:365)=0;
var_API(1:365)=0;
DOY(1:365)=0;

%total_mean_TA = mean(ta_observed_climatology(ta_observed_climatology > -100));%Global average high temperature
%total_mean_PET = mean(PET_observed_climatology(PET_observed_climatology > -100));%Global average high temperature
total_mean_TA = 288;%long term global average TA over land
total_mean_PET = 270;

if (API_model_flag == 3 || API_model_flag == 5) %make sure you don't encounter -999 in PET
    if (numel(PET_observed_climatology(PET_observed_climatology == -999)) > 0)
        total_mean_PET = 0;
        PET_observed(1:ist) = 0;
        PET_observed_climatology(1:365) = 0;
    end
end

% API coefficient definition

% sum for climatologies
DOY(1) = 365*((1+31)*0.002739726 - floor((1+31)*0.002739726));
for k=2:ist
    DOY(k) = 365*((k+31)*0.002739726 - floor((k+31)*0.002739726));
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
    API_model(k) = API_COEFF(k)*API_model(k-1) + rain_observed(k);
    if (API_model(k) > upper_bound_API); API_model(k) = upper_bound_API; end;

    if (sm_observed(k) > 0)
        sm_observed_DOY(round(DOY(k))) = sm_observed_DOY(round(DOY(k))) + sm_observed(k);
        sm2_observed_DOY(round(DOY(k))) = sm2_observed_DOY(round(DOY(k))) + sm_observed(k)*sm_observed(k);
        count_DOY_sm(round(DOY(k))) = count_DOY_sm(round(DOY(k))) + 1;
    end
   
    API_DOY(round(DOY(k))) = API_DOY(round(DOY(k)))+API_model(k);
    API2_DOY(round(DOY(k))) = API2_DOY(round(DOY(k)))+API_model(k)*API_model(k);
    count_DOY_API(round(DOY(k))) = count_DOY_API(round(DOY(k))) + 1;
end

total_sd_ratio = (sqrt(var(API_model))/sqrt(var(sm_observed((sm_observed >= 0)))));
total_sd_ratio_ERS = (sqrt(var(API_model))/sqrt(var(ERS_observed((ERS_observed >= 0)))));

total_mean_sm_observed = mean(sm_observed((sm_observed >= 0)));
total_var_sm_observed = var(sm_observed((sm_observed >= 0)));

total_mean_ERS_observed = mean(ERS_observed((ERS_observed >= 0)));
total_var_ERS_observed = var(ERS_observed((ERS_observed >= 0)));

total_mean_API = mean(API_model);

% average climatologies within 31-day moving windows
for k=1:365
    count_in_window_sm_observed=0;
    count_in_window_API=0;
    count_in_window_ERS_observed=0;
    for i=k-15:k+15
        i_prime=i;
        if (i<1); i_prime=i+365; end;
        if (i>365); i_prime=i-365; end;
        if (count_DOY_sm(i_prime) > 0)
            count_in_window_sm_observed = count_in_window_sm_observed + count_DOY_sm(i_prime);
            mean_sm_observed(k) = mean_sm_observed(k) + sm_observed_DOY(i_prime);
            var_sm_observed(k) = var_sm_observed(k) + sm2_observed_DOY(i_prime);
        end
       % if (count_DOY_ERS(i_prime) > 0)
       %     count_in_window_ERS_observed = count_in_window_ERS_observed + count_DOY_ERS(i_prime);
       %     mean_ERS_observed(k) = mean_ERS_observed(k) + ERS_observed_DOY(i_prime);
       %     var_ERS_observed(k) = var_ERS_observed(k) + ERS2_observed_DOY(i_prime);
       % end
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

    if (count_in_window_ERS_observed > 0)
        mean_ERS_observed(k)=mean_ERS_observed(k)/count_in_window_ERS_observed;
        var_ERS_observed(k)=(count_in_window_ERS_observed/(count_in_window_ERS_observed-1))*(var_ERS_observed(k)/count_in_window_ERS_observed - mean_ERS_observed(k)*mean_ERS_observed(k));
    else
        mean_ERS_observed(k) = total_mean_ERS_observed;
        var_ERS_observed(k) = total_var_ERS_observed;
    end

    mean_API(k)=mean_API(k)/count_in_window_API;
    var_API(k)=(count_in_window_API/(count_in_window_API-1))*(var_API(k)/count_in_window_API - mean_API(k)*mean_API(k));
end

if (location_flag == 0); [RMSE_API,RMSE_ERS,RMSE_AMSRE] = TC(API_model,mean_API,ERS_observed,mean_ERS_observed,sm_observed,mean_sm_observed,ist); end;%FOR ANOMALIES IN API SPACE

%converge_approach=1; %1) whiten,2) variance - using R in obs space
if (Q_fixed == 999)
    Q = 2000; %Initital condition only - start very large
    no_tune_flag = 0;
    P_inflation = P_inflation_fixed;
    converge_approach = 1;
else
    Q = Q_fixed;
    no_tune_flag = 1;  % To do variance tuning, change this to 0
    P_inflation = P_inflation_fixed;
    converge_approach = 2;
end;

logn_var = logn_var_constant;
converge_flag = 0;
converge_count = 0;

if (transform_flag == 1)
    RS_sort=sort(sm_observed((sm_observed >= 0)));
    API_sort=sort(transpose(API_model((sm_observed >= 0))));
end

for k=2:ist % rescaling observations and defining R

    if (location_flag > 0)
        if (EVI_observed(1+floor(k/30.5)) > 0.00) % every month pull a new EVI
            sqrt_R = 0.20*EVI_observed(1 + floor(k/30.5));
        else
            sqrt_R = 0.04;
        end
        R = sqrt_R^2;
        R_API(k) = R*(total_sd_ratio)^2;
    else
        R_API(k) = (RMSE_AMSRE)^2;
    end
    
    sm_observed_trans(k) = -1;
    ERS_observed_trans(k) = -1;

    if (transform_flag == 1 && sm_observed(k) >= 0); sm_observed_trans(k) = mean(API_sort((RS_sort == sm_observed(k))));end;
    if (transform_flag == 2 && sm_observed(k) >= 0); sm_observed_trans(k) = (sm_observed(k) - mean_sm_observed(round(DOY(k)))) * total_sd_ratio + mean_API(round(DOY(k))); end;
    if (transform_flag == 3 && sm_observed(k) >= 0); sm_observed_trans(k) = (sm_observed(k) - total_mean_sm_observed) * total_sd_ratio + total_mean_API; end;
    
    if (transform_flag == 4 && sm_observed(k) >= 0)
        delta_DOY = abs(DOY - DOY(k));
        delta_DOY(delta_DOY > 182.5) = 365 - delta_DOY(delta_DOY > 182.5);
        sm_observed_subset = sm_observed(abs(delta_DOY) <= 45);
        API_model_subset = API_model(abs(delta_DOY) <= 45);
        RS_sort=sort(sm_observed_subset((sm_observed_subset >= 0)));
        API_sort=sort(transpose(API_model_subset((sm_observed_subset) >= 0)));
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

while (converge_flag == 0)
    innovation_cross_sum = 0;
    count_updates = 0;
    innovation_last = 0;

    % rainfall correction with KF or RTS
    if (filter_flag==1 || filter_flag==5)
        for k=2:ist

            % Propagate API (Yixin)
            API_filter(k) = API_COEFF(k) * API_filter(k-1) + rain_observed(k);
            API_filter_f(k) = API_filter(k); %FOR RTS ONLY
            if (API_filter(k) > upper_bound_API); API_filter(k) = upper_bound_API; end;
            
            % Propagate error covariance matrix P (Yixin)
            P(k) = P(k-1)*API_COEFF(k)*API_COEFF(k) + Q + P_inflation*rain_observed(k)^2;
            Pf(k) = P(k); %FOR RTS ONLY             
            
            % Calculate gain K (Yixin)
            K(k) = P(k)/(P(k) + R_API(k));

            % Update API and P; if no sm measurement at this time point, skip (Yixin)
            if (sm_observed(k) < 0)% && ERS_observed (k) < -999999
                increment(k)=-999;
                innovation1(k)=-999;
            else
                count_updates = count_updates + 1;
                       
                innovation1(k)  = (sm_observed_trans(k) - (API_filter(k)))/sqrt(P(k) + R_API(k));
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                
                increment(k)= K(k)*(sm_observed_trans(k) - API_filter(k));

                API_filter(k) = API_filter(k) + K(k)*(sm_observed_trans(k) - API_filter(k));
                P(k) = (1 - K(k))*P(k);

                if (API_filter(k) > upper_bound_API); API_filter(k) = upper_bound_API; end;
            end
            increment_hold(k) = increment(k);

            if (filter_flag == 5 && k/window_size == floor(k/window_size))  %RTS filter
                RTS_API(k)=API_filter(k);
                for m=k-1:-1:k-window_size + 1
                    A = P(m)*API_COEFF(m)/Pf(m+1);
                    delta = A*(RTS_API(m+1)-API_filter_f(m+1));
                    RTS_API(m) = API_filter(m) + delta;
                    if (increment(m) ~= -999); increment(m) = increment(m) +  delta; end;
                    if (increment(m) == -999); increment(m) = delta; end;
                end
            end
        end
    end

    % rainfall correction with EnKF or EnKS
    if (filter_flag== 2 || filter_flag == 6)
        randn(10000);
        for k=2:ist
            
            % Propagate ensemble (Yixin)
            if (API_estimate_flag == 1)
                API_COEFF_E(k,:) = 0.95*API_COEFF_E(k-1,:) + 0.005*randn(1,NUMEN);
                API_COEFF_V(k,:) = ones*API_COEFF(k) +  API_COEFF_E(k,:);
            else
                API_COEFF_V(k,:) = ones*API_COEFF(k);
            end;

            mult_factor = exp(randn(1,NUMEN)*sqrt(log(logn_var + 1)) - log(logn_var + 1)/2);
            API_filter_EnKF(k,:) = API_COEFF_V(k,:).*API_filter_EnKF(k-1,:) + mult_factor*rain_observed(k) + sqrt(Q)*randn(1,NUMEN) + sqrt(P_inflation)*rain_observed(k)*randn(1,NUMEN);
            
            % Calculate gain K and update (if no sm observation, skip - Yixin)
            if (sm_observed(k) < 0)
                increment(k)=-999;
                innovation1(k)=-999;
            else
                % Calculate gain K
                P(k) = var(API_filter_EnKF(k,:));
                K(k) = P(k)/(P(k) + R_API(k));

                % EnSRF Implementation
                % K_EnSRF(k) = K(k)*(1 + sqrt(R_API(k)/(P(k) + R_API(k))))^-1;

                if (API_estimate_flag == 1)
                    hold = cov(API_filter_EnKF(k,:),API_COEFF_E(k,:));
                    CYM_API_COEFF(k) = hold(1,2);
                    K_API_COEFF(k) = CYM_API_COEFF(k)/(P(k) + R_API(k));
                else
                    K_API_COEFF(k) = 0;
                end

                count_updates=count_updates + 1;
                background = mean(API_filter_EnKF(k,:));
                innovation1(k)  = (sm_observed_trans(k) - background)/sqrt(P(k) + R_API(k));
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                increment(k) = K(k)*(sm_observed_trans(k) - background);  % mean increment (Yixin)
                hold_state = API_filter_EnKF(k,:);
                hold_perturbation = sqrt(R_API(k))*randn(1,NUMEN);

                %EnSRF - Square-root filter...have not noticed any advantage in using.
                %deltas = API_filter_EnKF(k,:) - background;
                %analysis = background + increment(k);
                %perturbations = deltas + K_EnSRF(k)*-deltas;
                %API_filter_EnKF(k,:) = analysis + perturbations;%En_SRF
                
                % Update ensemble
                API_filter_EnKF(k,:) = hold_state + K(k)*(sm_observed_trans(k) + hold_perturbation - hold_state);
                API_COEFF_E(k,:) = API_COEFF_E(k,:) + K_API_COEFF(k)*(sm_observed_trans(k) + hold_perturbation - hold_state);
            end

            increment_hold(k) = increment(k);

            if (filter_flag == 6 && k/window_size == floor(k/window_size))  %EnKS
                if (sm_observed_trans(k-1) >= 0) %this only turns on the EnKS if - on the second-to-last day of the window - you have an observation...not quite right
                    for m=k-2:-1:k-window_size + 1
                        CYM = cov(API_filter_EnKF(k,:),API_filter_EnKF(m,:));
                        K_EnKS  = CYM(1,2)/(var(API_filter_EnKF(k,:)) + R_API);
                        increment_EnKS(m) = K_EnKS*(sm_observed_trans(k) - background);
                        if (increment(m) ~= -999); increment(m) = increment(m) +  increment_EnKS(m); end;
                        if (increment(m) == -999); increment(m) = increment_EnKS(m); end;
                    end
                end
                if (sm_observed_trans(k) >= 0) %this only turns on the EnKS if - on the last day of the window - you have an observation...not quite right
                    for m=k-1:-1:k-window_size + 1
                        CYM = cov(API_filter_EnKF(k,:),API_filter_EnKF(m,:));
                        K_EnKS  = CYM(1,2)/(var(API_filter_EnKF(k,:)) + R_API);
                        increment_EnKS(m) = K_EnKS*(sm_observed_trans(k) - background);
                        if (increment(m) ~= -999); increment(m) = increment(m) +  increment_EnKS(m); end;
                        if (increment(m) == -999); increment(m) = increment_EnKS(m); end;
                    end
                end
            end
        end
    end

    % rainfall correction with direct insertion
    if (filter_flag==3)
        for k=2:ist
            API_filter(k) = API_COEFF(k) * API_filter(k-1) + rain_observed(k);
            %if (API_filter(k) > upper_bound_API); API_filter(k) = upper_bound_API; end;
            K(k) = 1.00;

            if (sm_observed(k) < 0)
                increment(k)=-999;
                innovation1(k)=-999;
            else
                count_updates = count_updates + 1;
                innovation1(k)  = (sm_observed_trans(k) - (API_filter(k)))/sqrt(P(k) + R_API);
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                increment(k)= K(k)*(sm_observed_trans(k) - (API_filter(k)));
                API_filter(k) = API_filter(k) + K(k)*(sm_observed_trans(k) - API_filter(k));
            end
        end
    end

    % rainfall correction with a particle filter
    if (filter_flag==4 || filter_flag == 7)
        randn(10000);
        for k=2:ist
            mult_factor = exp(randn(1,NUMEN)*sqrt(log(logn_var + 1)) - log(logn_var + 1)/2);
            rain = rain_observed(k);
            if (filter_flag == 4); API_filter_PART(k,:) = API_COEFF(k) * API_filter_PART(k-1,:) + sqrt(Q)*randn(1,NUMEN) + mult_factor*rain + sqrt(P_inflation)*rain_observed(k)*randn(1,NUMEN);end;
            if (filter_flag == 7); API_filter_PART(k,:) = API_COEFF(k) * API_filter_PART(k-1,:) + mult_factor*rain;end;
            bound_list = find(API_filter_PART(k,:) > upper_bound_API);
            API_filter_PART(k,bound_list) = upper_bound_API;
            background = mean(API_filter_PART(k,:));
            P(k) = var(API_filter_PART(k,:));

            if (sm_observed(k) < 0)
                increment(k)=-999;
                innovation1(k)=-999;
            else
                count_updates=count_updates + 1;

                eps = 1e-12;
                WEIGHT = normpdf(sm_observed_trans(k),API_filter_PART(k,:),sqrt(R_API(k)));
                if sum(WEIGHT)<eps
                    WEIGHT(1:NUMEN)=1/NUMEN;
                else
                    WEIGHT(1:NUMEN)=WEIGHT./sum(WEIGHT);
                end

                F = cumsum(WEIGHT);
                if abs(F(NUMEN) - 1) > eps
                    error('the weight vector should be normalized');
                end
                s = sort(rand(1,NUMEN));
                outIndex = zeros(1,NUMEN);
                m = 1;
                for i = 1:NUMEN
                    while F(m) < s(i)
                        m = m + 1;
                    end;
                    outIndex(i) = m;
                end;

                innovation1(k)  = (sm_observed_trans(k) - background)/sqrt(P(k) + R_API(k));
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                API_filter_PART(k,:) = API_filter_PART(k,outIndex);

                if (filter_flag == 7); increment(k) = (mult_factor*rain)*WEIGHT' - rain; end;
                if (filter_flag == 4); increment(k) = mean(API_filter_PART(k,:) - background); end;
                %if (CKF_flag == 1); increment(k) = increment(k)*(logn_var*rain_observed(k))/(logn_var*rain_observed(k) + Q); end;
            end
        end
    end

    innovation_mean = mean(innovation1((sm_observed >= 0)));
    innovation_var = var(innovation1((sm_observed >= 0)));
    innovation_ac = innovation_cross_sum/count_updates;
    innovation_lag1_covar = count_updates/(count_updates - 1) * (innovation_ac - innovation_mean*innovation_mean);
    innovation_lag1_corr = (innovation_lag1_covar/innovation_var);

    if (converge_approach == 1)
        if  (abs(innovation_lag1_corr) < 0.001 || converge_count > 25  || isnan(Q) == 1 || filter_flag == 3 || filter_flag == 7 || no_tune_flag == 1) %JUST USING THIS TO GENERATE "OLD" RESULTS...
            converge_flag = 1;
        else
            converge_count = converge_count + 1;
            if (converge_count == 1)
                Q_prime = Q;
                innovation_prime = innovation_lag1_corr;
                Q = 0;
                if (innovation_lag1_corr > 0) %If large initial Q cannot induce negative auto-correlation...stop
                    converge_flag = 1;
                end
            end

            if (converge_count == 2 && innovation_lag1_corr <= 0 && (filter_flag == 2 || filter_flag == 4 || filter_flag == 4))  %For ensemble-based approaches, if large Q=0 still leads to negetive auto-correlation...reduce logn_var
                converge_count = 1;
                logn_var = logn_var * 0.8;
            end

            if (converge_count >= 2)
                Q_hold = Q;
                power = 0.2; %This linearizes things - aids in convergence
                Q_trans = Q^power + (-innovation_lag1_corr) * (Q_prime^power - Q^power)/(innovation_prime-innovation_lag1_corr);
                Q = Q_trans^(1/power);

                if (Q < 0)
                    Q = Q_prime/4;
                end

                if (Q > 20000) %Protects against run-away convergence...
                    Q = 20000;
                    converge_flag = 1;
                end
                innovation_prime = innovation_lag1_corr;
                Q_prime = Q_hold;
            end
        end
    end

    if (converge_approach == 2)
        if  (abs(innovation_var-1) < 0.001 || converge_count > 25  || isnan(Q) == 1 || filter_flag == 3 || no_tune_flag == 1)
            converge_flag = 1;
        else
            converge_count = converge_count + 1;

            if (converge_count >= 2 && P_inflation < 0.001 && Q < 0.001 && innovation_var < 1); converge_flag = 1; end;

            if (converge_count == 1)
                Q_prime = Q;
                innovation_prime = innovation_var;
                Q = 0;
            end

            if (converge_count == 2 && innovation_var < 1 && (filter_flag == 2 || filter_flag == 4))
                converge_count = 0;
                logn_var = logn_var * 0.8;
                Q = 2000;
            end

            if (converge_count == 2 && innovation_var < 1 && (filter_flag == 1 || filter_flag == 5))
                converge_count = 0;
                P_inflation = P_inflation * 0.8;
                Q = 2000;
            end

            if (converge_count >= 2)
                Q_hold = Q;
                Q = Q + (-innovation_var+1) * (Q_prime - Q)/(innovation_prime-innovation_var);
                if (Q < 0)
                    Q = 0;
                end
                innovation_prime = innovation_var;
                Q_prime = Q_hold;
            end
        end
    end
end


% Sum up daily increments window each analysis window

increment_sum(1:floor(ist/window_size))=-999;
increment_sum_hold(1:floor(ist/window_size))=-999;

for k=2:floor(ist/window_size)
    for i=1:window_size
        sum_rain(k) = sum_rain(k) + rain_true((k-1)*window_size+ i);
        sum_rain_sp(k) = sum_rain_sp(k) + rain_observed((k-1)*window_size + i);
        sum_rain_indep(k) = sum_rain_indep(k) + rain_indep((k-1)*window_size + i);
        sum_rain_sp_hold(k) = sum_rain_sp_hold(k) + rain_observed_hold((k-1)*window_size + i);

        if (filter_flag <= 4 || filter_flag == 7)
            if (increment((k-1)*window_size + i) ~= -999)
                if (increment_sum(k) ~= -999); increment_sum(k) = increment_sum(k) + increment((k-1)*window_size + i);end;
                if (increment_sum(k) == -999); increment_sum(k) = increment((k-1)*window_size + i);end;
                increment_sum_hold(k) = increment_sum(k);
            end
        else %for smoothers
            if (increment_hold((k-1)*window_size + i) > -900)
                if (increment_sum(k) > -900); increment_sum(k) = increment_sum(k) + increment((k-1)*window_size + i);end;
                if (increment_sum(k) < -900); increment_sum(k) = increment((k-1)*window_size + i);end;
            end

            if (increment_hold((k-1)*window_size + i) > -900)
                if (increment_sum_hold(k) > -900); increment_sum_hold(k) = increment_sum_hold(k) + increment((k-1)*window_size + i);end;
                if (increment_sum_hold(k) < -900); increment_sum_hold(k) = increment((k-1)*window_size + i);end;
            end
        end
    end
end
end
