function [increment_sum,increment_sum_hold,sum_rain,sum_rain_sp,sum_rain_sp_hold,sum_rain_indep,increment_sum_ens, innovation1, innovation1_not_norm, rain_perturbed_sum_ens, rain_perturbed_ens] =...
    analysis(window_size,ist,filter_flag,transform_flag,API_model_flag,NUMEN,Q_fixed,P_inflation_fixed,...
    logn_var_constant,phi,bb,rain_observed,rain_observed_hold,rain_indep,rain_true,if_rescale,sep_sm_orbit,sma_observed,smd_observed,...
    ta_observed,ta_observed_climatology,PET_observed,PET_observed_climatology,EVI_observed,...
    API_mean,R_DQX, API_range, slope_parameter_API, time_step)

lag = 0.00; %Only used if API is varying seasonally
%API_estimate_flag = 0;

if (filter_flag ~= 1 || filter_flag ~=5)
    API_filter_EnKF(1:ist,1:NUMEN)=0;
    API_filter_EnKF_prior(1:ist,1:NUMEN)=0;
    API_filter_PART(1:ist,1:NUMEN)=0;
    API_COEFF_E(1:ist,1:NUMEN)=0.00;
    API_COEFF_V(1:ist,1:NUMEN)=0.80;
    ones(1:NUMEN)=1;
end

API_filter_f(1:ist)=0;
RTS_API(1:ist)=0;
API_COEFF_EF(1:ist)=0;
API_filter(1:ist)=0;
increment(1:ist)=0;
increment_ens(1:ist, 1:NUMEN) = 0;
rain_perturbed_ens(1:ist, 1:NUMEN) = 0;
innovation1(1:ist)=0;
innovation1_not_norm(1:ist)=0;

P(1:ist)=0;
Pf(1:ist)=0;
K(1:ist)=0;
CYM_API_COEFF(1:ist) = 0;
K_API_COEFF(1:ist) = 0;
sum_rain(1:floor(ist/window_size)) = 0;
sum_rain_sp(1:floor(ist/window_size)) = 0;
sum_rain_sp_hold(1:floor(ist/window_size)) = 0;
sum_rain_indep(1:floor(ist/window_size)) = 0;
increment_sum(1:floor(ist/window_size)) = 0;
increment_sum_ens(1:floor(ist/window_size), 1:NUMEN)=0;
rain_perturbed_sum_ens(1:floor(ist/window_size), 1:NUMEN)=0;

% ----- Rescale sm observations ----- %
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

% If sep_sm_orbit = 0, merge ascending and descending SM below to
% a single product, and then rescale
if sep_sm_orbit == 0
    % Merge products
    sm_observed(1:ist) = nan;
    for k=1:ist
        if (~isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
            sm_observed(k) =  0.5*(smd_observed(k) + sma_observed(k));
        end;
        if (isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
            sm_observed(k) =  smd_observed(k); end;
        if (~isnan(sma_observed(k)) && isnan(smd_observed(k)) );
            sm_observed(k) =  sma_observed(k); end;
    end
    % Rescale
    [sm_observed_trans, R_API, API_COEFF, API_model] = rescale(sm_observed, time_step, ...
        transform_flag, API_model_flag, ist, rain_observed, API_mean, API_range, lag, ...
        slope_parameter_API, ta_observed_climatology, ...
        PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
        EVI_observed, R_DQX);
% If sep_sm_orbit = 1, rescale ascending & descending sm observations
% separately; sm_observed only serves as an indicator for update timesteps
else
    % Rescale ascending & descending separately
    [sma_observed_trans, R_API_a, API_COEFF_a, API_model] = rescale(sma_observed, time_step, ...
        transform_flag, API_model_flag, ist, rain_observed, API_mean, API_range, lag, ...
        slope_parameter_API, ta_observed_climatology, ...
        PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
        EVI_observed, R_DQX);
    [smd_observed_trans, R_API_d, API_COEFF_d, API_model] = rescale(smd_observed, time_step, ...
        transform_flag, API_model_flag, ist, rain_observed, API_mean, API_range, lag, ...
        slope_parameter_API, ta_observed_climatology, ...
        PET_observed_climatology, total_mean_TA, total_mean_PET, bb, ...
        EVI_observed, R_DQX);
    % Put rescaled ascending & desceinding sm together
    sm_observed_trans(1:ist) = nan;
    R_API(1:ist) = 0;
    API_COEFF(1:ist) = API_mean;
    sm_observed(1:ist) = nan;
    for k=1:ist
        if (~isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
            fprintf('Error: When sep_sm_orbit, ascending and descending products cannot appear on the same timestep!');
            exit(1);
        end;
        if (isnan(sma_observed(k)) && ~isnan(smd_observed(k)) );
            sm_observed_trans(k) =  smd_observed_trans(k);
            R_API(k) = R_API_d(k);
            API_COEFF(k) = API_COEFF_d(k);
            sm_observed(k) = smd_observed(k);
        end;
        if (~isnan(sma_observed(k)) && isnan(smd_observed(k)) );
            sm_observed_trans(k) =  sma_observed_trans(k);
            R_API(k) = R_API_a(k);
            API_COEFF(k) = API_COEFF_a(k);
            sm_observed(k) = sma_observed(k);
        end;
    end
end

% If not rescale, reset SM and R to unscaled values
if if_rescale == 0
    R_API(:) = R_DQX.^2;
    sm_observed_trans = sm_observed;
end

% ----- Run filter ----- %
% innovation whitening....this is no longer being supported.....
%     Q = 2000; %Initital condition only - start very large
%     no_tune_flag = 0;
%     P_inflation = P_inflation_fixed;
%     converge_approach = 1;

Q = Q_fixed;
P_inflation = P_inflation_fixed;
converge_approach = 2;
if (Q==999)
    no_tune_flag = 0;
    Q=900; %first guess for Q
else
    no_tune_flag = 1;
end;

logn_var = logn_var_constant;
converge_flag = 0;
converge_count = 0;

% Initialize ensemble perturbed rainfall to be observed rainfall
for (i = 1:NUMEN)
    rain_perturbed_ens(:, i) = rain_observed;
end

% Identify update days
if (filter_flag == 5 || filter_flag == 6) 
    update_days = find(~isnan(sm_observed_trans));
end

while (converge_flag == 0)
    innovation_cross_sum = 0;
    count_updates = 0;
    innovation_last = 0;
    
    % rainfall correction with EKF and ERTS
    if (filter_flag==1 || filter_flag==5)
        for k=2:ist
            
            % Propagate API (Yixin)
            API_filter(k) = sign(API_filter(k-1))*(API_COEFF(k)-1)*abs(API_filter(k-1))^bb + API_filter(k-1)  + rain_observed(k);
            
            API_filter_f(k) = API_filter(k);
            if (API_filter(k-1) ~= 0.0)
                API_COEFF_EF(k) = (API_COEFF(k)-1)*bb*abs(API_filter(k-1))^(bb-1) + 1;
            else
                API_COEFF_EF(k) = (API_COEFF(k)-1);
            end
            
            % Propagate error covariance matrix P (Yixin)
            P(k) = P(k-1)*API_COEFF_EF(k)*API_COEFF_EF(k) + Q + P_inflation*rain_observed(k)^2;
            Pf(k) = P(k);
            
            % Calculate gain K;
            K(k) = P(k)/(P(k) + R_API(k));
            
            % Update API and P (Yixin)
            % If filter_flag == 1 and no sm measurement at this time point,
            % skip;
            % If filter_flag == 5 and no sm measurement at this time point,
            % go back in time until last update and update the gaps
            if (isnan(sm_observed(k)))
                increment(k)=-999;
                innovation1(k)=-999;
                innovation1_not_norm(k)=-999;
            else
                count_updates = count_updates + 1;
                innovation1(k)  = (sm_observed_trans(k) - (API_filter(k)))/sqrt(P(k) + R_API(k));
                innovation1_not_norm(k)  = sm_observed_trans(k) - (API_filter(k));
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                increment(k)= K(k)*(sm_observed_trans(k) - API_filter(k));
                API_filter(k) = API_filter(k) + K(k)*(sm_observed_trans(k) - API_filter(k));
                P(k) = (1 - K(k))*P(k);
                
                %RTS filter
                if (filter_flag == 5)
                    temp=update_days-k;
                    temp(temp >= 0)=nan;
                    [junk,index]=max(temp);
                    last_update = update_days(index);
                    RTS_API(k)=API_filter(k);
                    
                    %Go back in time until last update
                    for m=k-1:-1:last_update+1
                        A = P(m)*API_COEFF_EF(m)/Pf(m+1);
                        delta = A*(RTS_API(m+1)-API_filter_f(m+1));
                        RTS_API(m) = API_filter(m) + delta;
                        if (increment(m) > -999)
                            fprintf (1,'problem %d %f \n',k,increment(m));
                        end
                        increment(m) = delta;
                    end
                end
            end
        end
    end
    
    % rainfall correction with EnKF or EnKS
    if (filter_flag== 2 || filter_flag == 6)
        % Generate mult_factor time series for each ensemble member
        mult_factor(1:ist, 1:NUMEN) = 0;  % [time, ens]
        for e=1:NUMEN
            mult_factor(:, e) = generate_prec_lognormal_multiplier(...
                logn_var, phi, ist);
        end
        
        for k=2:ist
            
            % Propagate ensemble and add state perturbation (Yixin)
            API_COEFF_V(k,:) = ones*API_COEFF(k);
            rain_perturbed_ens(k, :) = mult_factor(k, :) * rain_observed(k);
            temp_total(1:NUMEN)=nan;
            for e=1:NUMEN
                temp_total(e) = API_short(API_filter_EnKF(k-1,e),API_COEFF_V(k,e),bb,1);
%                  Implicit_test = sign(temp_total(e))*(API_COEFF_V(k,e)-1)*abs(temp_total(e))^bb + API_filter_EnKF(k-1,e);
%                  if (abs(temp_total(e)-Implicit_test) >= 5)
%                       temp_total(e) = API_short(API_filter_EnKF(k-1,e),API_COEFF_V(k,e),bb,24);
%                  end
            end
            API_filter_EnKF(k,:) = temp_total + rain_perturbed_ens(k, :) + sqrt(Q)*randn(1,NUMEN) + sqrt(P_inflation)*rain_observed(k)*randn(1,NUMEN);         
            API_filter_EnKF_prior(k,:) = API_filter_EnKF(k,:);
            
            % Calculate gain K and update (Yixin)
            % If filter_flag == 2 and no sm measurement at this time point,
            % skip;
            % If filter_flag == 6 and no sm measurement at this time point,
            % go back in time until last update and update the gaps
            if (isnan(sm_observed(k)))
                increment(k)=-999;
                innovation1(k)=-999;
                innovation1_not_norm(k)=-999;
            else
                % Calculate gain K
                P(k) = var(API_filter_EnKF(k,:));
                K(k) = P(k)/(P(k) + R_API(k));
                
                count_updates=count_updates + 1;
                background = mean(API_filter_EnKF(k,:));
                hold_state = API_filter_EnKF(k,:);
                hold_perturbation = sqrt(R_API(k))*randn(1,NUMEN);
                
                innovation1(k)  = (sm_observed_trans(k) - background)/sqrt(P(k) + R_API(k));
                innovation1_not_norm(k)  = sm_observed_trans(k) - background;
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                
                increment(k) = K(k)*(sm_observed_trans(k) - background);
                increment_ens(k,:) = K(k)*(sm_observed_trans(k) + hold_perturbation - API_filter_EnKF(k,:));
                
                API_filter_EnKF(k,:) = hold_state + K(k)*(sm_observed_trans(k) + hold_perturbation - hold_state);
                
                if (filter_flag == 6)  %EnKS gap filling
                    % find last update time step
                    temp=update_days-k;
                    temp(temp >= 0)=nan;
                    [junk,index]=max(temp);
                    last_update = update_days(index);                 
                    % Set upper limit to the earliest timestep to fill to
                    % 10 days
                    max_steps_to_fill = 10 * (24 / time_step);
                    if (k - last_update > max_steps_to_fill)
                        last_update = k - max_steps_to_fill;
                    end
                    for m=k-1:-1:last_update + 1
                        % Use priors to generated K_EnKF
                        CYM = cov(API_filter_EnKF_prior(k,:),API_filter_EnKF_prior(m,:));
                        K_EnKS  = CYM(1,2)/(var(API_filter_EnKF_prior(k,:)) + R_API(k));
                        % Update - deterministic
                        increment(m) = K_EnKS*(sm_observed_trans(k) - background);                     
                        % Update - ensemble
                        increment_ens(m,:) = K_EnKS*(sm_observed_trans(k) + hold_perturbation - hold_state);                                        
                    end
                end
            end
        end
    end
    
    % rainfall correction with direct insertion
    if (filter_flag==3)
        for k=2:ist
            API_filter(k) = sign(API_filter(k-1))*(API_COEFF(k)-1)*abs(API_filter(k-1))^bb + API_filter(k-1)  + rain_observed(k);
            K(k) = 1.00;
            if (isnan(sm_observed(k)))
                increment(k)=-999;
                innovation1(k)=-999;
                innovation1_not_norm(k)=-999;
            else
                count_updates = count_updates + 1;
                innovation1(k)  = (sm_observed_trans(k) - (API_filter(k)))/sqrt(P(k) + R_API(k));
                innovation1_not_norm(k)  = sm_observed_trans(k) - API_filter(k);
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                increment(k)= K(k)*(sm_observed_trans(k) - (API_filter(k)));
                API_filter(k) = API_filter(k) + K(k)*(sm_observed_trans(k) - API_filter(k));
            end
        end
    end
    
    % rainfall correction with a particle filter
    % 4 = PF, 7 = direct rainfall bayesian updating
    % filtering only...no smoothing back correction...has not been
    % completely de-bugged...use with caution
    if (filter_flag==4 || filter_flag == 7)
        mult_factor = generate_prec_lognormal_multiplier(...
            logn_var, phi, ist);
        for k=2:ist
            add_factor = randn(1,NUMEN);
            rain = rain_observed(k);
            
            if (filter_flag == 4)
                for e=1:NUMEN
                    temp_total(e) = API_short(API_filter_PART(k-1,e),API_COEFF_V(k,e),bb,1);
                    %Implicit_test = sign(temp_total(e))*(API_COEFF_V(k,e)-1)*abs(temp_total(e))^bb + API_filter_PART(k-1,e);
                   %if (abs(temp_total(e)-Implicit_test) >= 5)
                   %      temp_total(e) = API_short(API_filter_PART(k-1,e),API_COEFF_V(k,e),bb,24);
                   %end
                end
                API_filter_PART(k,:) = temp_total + mult_factor(k)*rain_observed(k) + sqrt(Q)*randn(1,NUMEN) + sqrt(P_inflation)*rain_observed(k)*randn(1,NUMEN);
                
                %older
                %temp = (API_COEFF(k)-1).*abs(API_filter_PART(k-1,:)).^bb;
                %temp_sign = sign(API_filter_PART(k-1,:));
                %temp_total = temp_sign.*temp;
                %API_filter_PART(k,:) = temp_total + API_filter_PART(k-1,:) + mult_factor*rain_observed(k) + sqrt(Q)*randn(1,NUMEN) + sqrt(P_inflation)*rain_observed(k)*randn(1,NUMEN);
         
                %even older
                %API_filter_PART(k,:) = API_COEFF(k) * API_filter_PART(k-1,:).^bb + sqrt(Q)*randn(1,NUMEN) + mult_factor*rain + sqrt(P_inflation)*rain_observed(k)*randn(1,NUMEN);
            else
                for e=1:NUMEN
                    temp_total(e) = API_short(API_filter_PART(k-1,e),API_COEFF_V(k,e),bb,1);
                    %Implicit_test = sign(temp_total(e))*(API_COEFF_V(k,e)-1)*abs(temp_total(e))^bb + API_filter_PART(k-1,e);
                    %if (abs(temp_total(e)-Implicit_test) >= 5)
                    %     temp_total(e) = API_short(API_filter_PART(k-1,e),API_COEFF_V(k,e),bb,24);
                    %end
                end
                API_filter_PART(k,:) = temp_total + mult_factor(k)*rain + add_factor*sqrt(Q);
                
                %older
                %temp = (API_COEFF(k)-1).*abs(API_filter_PART(k-1,:)).^bb;
                %temp_sign = sign(API_filter_PART(k-1,:));
                %temp_total = temp_sign.*temp;
                %API_filter_PART(k,:) = temp_total + API_filter_PART(k-1,:) + mult_factor*rain + add_factor*sqrt(Q);
                
                %even older
                %API_filter_PART(k,:) = API_COEFF(k) * API_filter_PART(k-1,:).^bb + mult_factor*rain;
            end;
            
            background = mean(API_filter_PART(k,:));
            P(k) = var(API_filter_PART(k,:));
            if (isnan(sm_observed(k)))
                increment(k)=-999;
                innovation1(k)=-999;
                innovation1_not_norm(k)=-999;
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
                innovation1_not_norm(k)  = sm_observed_trans(k) - background;
                innovation_cross_sum = innovation_cross_sum + innovation1(k)*innovation_last;
                innovation_last = innovation1(k);
                
                API_filter_PART(k,:) = API_filter_PART(k,outIndex);
                if (filter_flag == 7); increment(k) = (mult_factor(k)*rain + add_factor*sqrt(Q))*WEIGHT' - rain; end;
                if (filter_flag == 4); increment(k) = mean(API_filter_PART(k,:) - background); end;
            end
        end
    end
    
    if (imag(mean(API_filter)) ~= 0)
        fprintf (1,'imaginary number problem \n');
    end
    
    %THIS HELPS SOMETIMES
    %increment(innovation1 < -2 & innovation1 > -999) = -999;
    
    innovation_mean = mean(innovation1((~isnan(sm_observed))));
    innovation_var = var(innovation1((~isnan(sm_observed))));
    innovation_ac = innovation_cross_sum/count_updates;
    innovation_lag1_covar = count_updates/(count_updates - 1) * (innovation_ac - innovation_mean*innovation_mean);
    innovation_lag1_corr = (innovation_lag1_covar/innovation_var);
    
    % CONVERGENCE
    % fprintf (1','%f %f %f %f %f \n',Q,P_inflation,innovation_lag1_corr,innovation_var,converge_count+1);
    if (converge_approach == 1) % Innovation whitening...no longer being supported
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
    
    if (converge_approach == 2) %used fixed R and then tune variance
        if  (abs(innovation_var-1) < 0.001 || converge_count > 25  || isnan(Q) == 1 || filter_flag == 3 || no_tune_flag == 1)
            converge_flag = 1;
        else
            converge_count = converge_count + 1;          
            if (converge_count >= 2 && P_inflation < 0.1 && Q < 0.1 && innovation_var < 1); converge_flag = 1; end;
            
            if (converge_count == 1)
                Q_prime = Q;
                innovation_prime = innovation_var;
                Q = 0;
            end
            % Sometimes even very large Q can cause innovation_var < 1...if so then adjust other error pararameters and start again.
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

% sum up to window_size
% Note...this could be moved out of analysis
% Yixin, this will have to be changed to accomodate increment ensembles
increment_sum(1:floor(ist/window_size))=-999;
increment_sum_hold(1:floor(ist/window_size))=-999;
for k=2:floor(ist/window_size)
    for i=1:window_size
        sum_rain(k) = sum_rain(k) + rain_true((k-1)*window_size+ i);
        sum_rain_sp(k) = sum_rain_sp(k) + rain_observed((k-1)*window_size + i);
        sum_rain_indep(k) = sum_rain_indep(k) + rain_indep((k-1)*window_size + i);
        sum_rain_sp_hold(k) = sum_rain_sp_hold(k) + rain_observed_hold((k-1)*window_size + i);
        rain_perturbed_sum_ens(k, :) = rain_perturbed_sum_ens(k, :) + rain_perturbed_ens(((k-1)*window_size + i), :);
        
        if (increment((k-1)*window_size + i) ~= -999)
            if (increment_sum(k) ~= -999)
                increment_sum(k) = increment_sum(k) + increment((k-1)*window_size + i);
                increment_sum_ens(k, :) = increment_sum_ens(k, :) + increment_ens((k-1)*window_size + i, :);
            end
            if (increment_sum(k) == -999)
                increment_sum(k) = increment((k-1)*window_size + i);
                increment_sum_ens(k, :) = increment_ens((k-1)*window_size + i, :);
            end
            increment_sum_hold(k) = increment_sum(k);
        end
    end
end

end
