function SMART(filter_flag,transform_flag,API_model_flag,lambda_flag,NUMEN,Q_fixed,P_inflation,upper_bound_API,logn_var,slope_parameter_API,location_flag,window_size)

%Argument definitions:
%WARNING...some of these choice are not fully implemented
%filter_flag 1)KF, 2)EnKF, 3)DI, 4)PART, 5)RTS, 6) EnKS, 7) PART - DIRECT RAINFALL
%transform_flag 1) CDF, 2) seasonal 1&2, 3) bias 1&2, 4) seasonal CDF 
%API_model_flag 0) static 1) simple sine model, 2) Ta climatology, 3) PET climatologoy, 4) Ta-variation, 5) PET variation
%lambda_flag 0) optimized, 1) PERSIANN, 2) 0.60 
%NUMEN - number of ensembles used in EnKF or EnKS analysis...not used if filter_flag  = 1 or 3
%Q_fixed - if = 999 than whiten tune, otherwise it sets Q
%P_inflation
%logn_var...not sued if filter_flag = 1 or 3....setting to zero means all rainfall error is additive
%slope parameter API - not used if API_model_flag = 0
%location flag 0) CONUS, 1) AMMA, 2) Global 3) Australia 31 4) Australia 240 5) Australia, 0.25-degree continental
%window size - time scale at which rainfall correction is applied 3 to 10 days is recommended

% Turn on/off diagnostic data dump
dump_flag = 0;

%% Australia - 0.25 Degree Continental Scale
if (location_flag == 5)
    load SMART_Input_SMOS_RE02_20100101_20131231.mat;
    dnum1 = dnum; % Save date number of input in a new dnum1
    numpixels = numel(lidx); % Number of land pixels of 0.25-deg Australia = 11125
    
    % Time period
    % Start
    yyyy1 = 2010;
    mm1 = 01;
    dd1 = 01;
    % End
    yyyy2 = 2013;
    mm2 = 12;
    dd2 = 31;
    
    dnum1 = datenum(yyyy1,mm1,dd1);
    dnum2 = datenum(yyyy2,mm2,dd2);
    dnum = (dnum1:dnum2);
    nd = numel(dnum);
    ist = nd;
    window_ist = floor(ist/window_size);
    
    RAIN_1 = AUS25KM_3B42RT';%Satellite-based precipitation
    RAIN_2 = AUS25KM_3B42'; %Calibration target
    RAIN_3 = AUS25KM_AWAP'; %Used as benchmark
    SMOBS_A = AUS25KM_SMOS_A'; %Soil Moisture - Ascending
    SMOBS_D = AUS25KM_SMOS_D'; %Soil Moisture - Descending
    
    clear AUS25KM_3B42RT AUS25KM_3B42 AUS25KM_AWAP AUS25KM_SMOS_A AUS25KM_SMOS_B;
    
    RAIN_1(find(isnan(RAIN_1)))=-999;
    RAIN_2(find(isnan(RAIN_2)))=-999;
    RAIN_3(find(isnan(RAIN_3)))=-999;   
    SMOBS_A(find(isnan(SMOBS_A)))=-999;
    SMOBS_D(find(isnan(SMOBS_D)))=-999;
    
end
%% 1. SMOS RE02 ASC-DSC composite

% Initialize corrected rainfall
RAIN_SMART_SMOS = zeros(window_ist,1);

% Summary statistics
outfn = ['SMART_AUS_SMOS_ASCDSC_' num2str(window_size) 'd.txt'];
output=fopen(outfn,'w');

for j=1:numpixels %space loop

    rain_observed = RAIN_1(:,j); %Satellite rainfall here
    rain_indep = RAIN_2(:,j); % Independent rainfall here/calibration target
    rain_true = RAIN_3(:,j); %RAIN_2(:,j); % AWAP rainfall
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
    if (water_fraction < 25 && sum(rain_indep(rain_indep > 0))...
            > 20 && sum(rain_true(rain_true > 0)) > 20 ...
            && sum(rain_observed(rain_observed > 0)) > 20 ...
            && sqrt(var(sma_observed((sma_observed >=0))))...
            > 0*1e-3 && sqrt(var(smd_observed((smd_observed >=0)))) > 0*1e-3) %&& sqrt(var(ERSa_observed((ERSa_observed >=0)))) > 0*1e-3 && sqrt(var(ERSd_observed((ERSd_observed >=0)))) > 0*1e-3)

        rain_observed_hold = rain_observed;
        rain_observed_hold(rain_observed_hold < 0) = -99999; % set to -9999 to throw out windows with missing days (STANDARD)....set to 0 to keep them in the analysis with a default value of zero.

        rain_observed(rain_observed < 0) = 0; %for the time series feed into the KF...get rid of missing rain data...dataset must be complete (no missing values)
        rain_true(rain_true < 0) = -99999; % all calibration/evaluation is based on windows with no missing indep or true rainfall data
        rain_indep(rain_indep < 0) = -99999; % all calibration/evaluation is based on windows with no missing indep or true rainfall data 
        rain_indep=rain_indep*mean(rain_observed_hold(rain_observed_hold >= 0))/mean(rain_indep(rain_indep >= 0)); %make sure RS precipitation products have same mean
                
        sm_observed(1:ist) = -1;
        ERS_observed(1:ist) = -1;
        
        % Naive merging ascending and descending SM below to a daily product
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
            analysis(window_size,ist,filter_flag,transform_flag,API_model_flag,NUMEN,Q_fixed,P_inflation,...
            logn_var,upper_bound_API,rain_observed,rain_observed_hold,rain_indep,rain_true,sm_observed,ERS_observed,...
            ta_observed,ta_observed_climatology,PET_observed,PET_observed_climatology,EVI_observed,slope_parameter_API,location_flag);
        
        % Apply increments to correct rainfall...time loop is in this
        % function
        [sum_rain_corrected,optimized_fraction] = ...
            correction(increment_sum,increment_sum_hold,sum_rain_sp,sum_rain_sp2,lambda_flag);
     
        % This is the corrected rainfall...below we evaluate it via
        % comparisons to an independent rainfall product
        RAIN_SMART_SMOS(:,j) = sum_rain_corrected(:);
        
        mask_sum_rain = sum_rain >= 0; %need benchmark to be there
        mask_sum_rain_sp = sum_rain_sp_hold >= 0; %need sat precip to be there (at least one)...
        mask_sm = increment_sum_hold ~= -999; %need sat sm to be there (at least one)...      
        mask_time_total = mask_sm.*mask_sum_rain_sp; %create mask for both sat sm and sat precip present
        mask_time_total = mask_time_total.*mask_sum_rain; %create final total mask for also masking for the benchmark to be present            
        sum_rain_subset = sum_rain(mask_time_total > 0 ); %apply mask
        sum_rain_sp_subset = sum_rain_sp(mask_time_total > 0); %apply mask
        sum_rain_corrected_subset = sum_rain_corrected(mask_time_total > 0); %apply mask
        
        hold1=mean(sum_rain_corrected_subset);
        hold2=mean(sum_rain_sp_subset);
        sum_rain_corrected_subset=sum_rain_corrected_subset.*(hold2/hold1); %STANDARD INCLUDES THIS...need this b/c of bias that develops..so we force no mean change from uncorrected....this should probably be taken into account when calibrating lambda....

        %TO DUMP ALL QUALIFYING N_DAY PERIODS FOR DIAGNOSTIC ANALYSIS...see "dump flag" set above
        if (dump_flag  == 1)
            for k=1:numel(sum_rain_subset)
                fprintf(output,'%f %f %f \n',sum_rain_subset(k),sum_rain_sp_subset(k),sum_rain_corrected_subset(k));
            end
        end
        
        A=corrcoef(sum_rain_subset,sum_rain_sp_subset);
        if numel(A)==1
            A=[A,0;0,A];
        end
        B=corrcoef(sum_rain_subset,sum_rain_corrected_subset);
        if numel(B)==1
            B=[B,0;0,B]; 
        end
        C=sqrt(mean((sum_rain_subset-sum_rain_sp_subset).^2));
        D=sqrt(mean((sum_rain_subset-sum_rain_corrected_subset).^2));
        E=mean(sum_rain_subset);
        F=mean(sum_rain_sp_subset);
        G=mean(sum_rain_corrected_subset);
        H=var(sum_rain_subset);
        I=var(sum_rain_sp_subset);
        J=var(sum_rain_corrected_subset);

        % Define rain event thresholds....
	    %sum_rain_nonzero_sort = sort(sum_rain_sp(sum_rain_sp > 2));  % USED THIS IN OLD RUNS
        sum_rain_nonzero_sort = sort(sum_rain_subset(sum_rain_subset > 2)); %THIS IS STANDARD 
        number_total = numel(sum_rain_nonzero_sort);

        FAR_corr(1:12)=0;
        POD_corr(1:12)=0;
        CSI_corr(1:12)=0;
        HR_corr(1:12)=0;
        E_CSI_corr(1:12)=0;
        FAR(1:12)=0;
        POD(1:12)=0;
        CSI(1:12)=0;
        HR(1:12)=0;
        E_CSI(1:12)=0;
        event_threshold(1:12)=0;
        
        %HARDWIRE TO SHUT OFF TREND OUTPUT
        trend_true = 0;
        trend_sp = 0;
        trend_corrected = 0;

        percentile(1)=0.00;
        percentile(2)=0.10;
        percentile(3)=0.20;
        percentile(4)=0.30;
        percentile(5)=0.40;
        percentile(6)=0.50;
        percentile(7)=0.60;
        percentile(8)=0.70;
        percentile(9)=0.80;
        percentile(10)=0.90;
        percentile(11)=0.95;
        percentile(12)=0.99;

        % THIS IS THE ACTUAL METRIC CALCULATION
        if (numel(sum_rain_nonzero_sort) > 0)
            for k=1:12
                event_threshold(k) =  sum_rain_nonzero_sort(floor(number_total*percentile(k))+1);
                floor(number_total*percentile(k))+1;
                number_hit_corr =   numel(find(sum_rain_corrected_subset((sum_rain_subset >= event_threshold(k))) >= event_threshold(k)));
                number_false_corr = numel(find(sum_rain_corrected_subset((sum_rain_subset < event_threshold(k)))  >= event_threshold(k)));
                number_miss_corr =  numel(find(sum_rain_corrected_subset((sum_rain_subset >= event_threshold(k)))  < event_threshold(k)));
                number_hit_no_corr = numel(find(sum_rain_corrected_subset((sum_rain_subset < event_threshold(k)))  < event_threshold(k)));

                FAR_corr(k) =       number_false_corr/(number_hit_corr + number_false_corr);
                if (isnan(FAR_corr(k))); FAR_corr(k) = 0; end; %STANDARD
                POD_corr(k) =       number_hit_corr/(number_miss_corr + number_hit_corr);
                CSI_corr(k) =       number_hit_corr/(number_miss_corr + number_false_corr + number_hit_corr);
                HR_corr(k) =        (number_hit_corr + number_hit_no_corr)/(numel(sum_rain_subset));
                A_R_corr =          (number_hit_corr + number_false_corr)*(number_hit_corr+number_miss_corr) / numel(sum_rain_subset);
                E_CSI_corr(k) =     (number_hit_corr - A_R_corr)/(number_miss_corr + number_false_corr + number_hit_corr - A_R_corr);

                number_hit =        numel(find(sum_rain_sp_subset((sum_rain_subset >= event_threshold(k))) >= event_threshold(k)));
                number_false =      numel(find(sum_rain_sp_subset((sum_rain_subset < event_threshold(k))) >= event_threshold(k)));
                number_miss =       numel(find(sum_rain_sp_subset((sum_rain_subset >= event_threshold(k))) < event_threshold(k)));
                number_hit_no =     numel(find(sum_rain_sp_subset((sum_rain_subset < event_threshold(k))) < event_threshold(k)));

                FAR(k) =            number_false/(number_hit + number_false);
                if (isnan(FAR(k))); FAR(k) = 0; end; %STANDARD
                POD(k) =            number_hit/(number_miss + number_hit);
                CSI(k) =            number_hit/(number_miss + number_false + number_hit);
                HR(k) =             (number_hit + number_hit_no)/(numel(sum_rain_subset));
                A_R =               (number_hit + number_false)*(number_hit+number_miss) / numel(sum_rain_subset);
                E_CSI(k) =          (number_hit - A_R)/(number_miss + number_false + number_hit - A_R);
            end
        else
            for k=1:12
                FAR(k) = 0;
                POD(k) = 0;
                CSI(k) = 0;
                FAR_corr(k) = 0;
                POD_corr(k) = 0;
                CSI_corr(k) = 0;
            end
        end

        CORR_API_SM=0;
        % output to file
        if (dump_flag == 0)
            %fprintf(output,'%f %f %f %f ',A(1,2)^2,B(1,2)^2,C,D);
            fprintf(output,'%f %f %f %f ',A(1,2)^2,B(1,2)^2,C,optimized_fraction);
            for k=1:12
                fprintf(output, '%f %f %f %f %f %f ',FAR(k),FAR_corr(k),POD(k),POD_corr(k),CSI(k),CSI_corr(k));
            end
            fprintf(output,'%f %f %f %f %f %f %f %f %f %f %f %f \n',E,F,G,H,I,J,optimized_fraction,CORR_API_SM,mean(EVI_observed),trend_true,trend_sp,trend_corrected);
            
            % output to screen
            fprintf(1,'%f %f %f %f \n',A(1,2),B(1,2),C,D);
        end
   else
       %output to file
       if (dump_flag == 0)
           fprintf(output,'-1 -1 -1 -1 ');
           for k=1:12
               fprintf(output, '-1 -1 -1 -1 -1 -1 ');
           end
           fprintf(output,'-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 \n');
           
           fprintf(1,'-1 -1 -1 -1 -1 \n ');
       end
    end
   %disp(['############ AMSR-E ############# j = ' num2str(j)]);  
end

fclose(output);

