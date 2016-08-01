
%To apply exact approach described in the original SMART paper
SMART(1,2,0,0,200,3,5,99999,0,0.01,5,5)
%last argument = time scale of correction (here = 5 days)

%To match Australian analysis (JGR 2016)
SMART(1,3,0,0,200,3,5,99999,0,0.01,5,5)
% Only change is to use a simpler rescaling (to match long-term - and not season 1st and 2nd statistical moments
% Speeds the code up and may (actually) slightly improve results

%For EnKF implementation (for ensemble analysis)
SMART(2,3,0,0,200,3,5,99999,0,0.01,5,5)
%In the EnKF you can also add multiplicative error by making logn_var
%non-zero...I am NOT doing that here...unsure of impact
%NUMEN sets number of ensemble in this case 

%Argument list:
%filter_flag 1)KF, 2)EnKF, 3)DI, 4)PART, 5)RTS, 6) EnKS, 7) PART - DIRECT RAINFALL
%transform_flag 1) CDF, 2) seasonal 1&2, 3) long-term 1&2, 4) seasonal CDF 
%API_model_flag 0) static 1) simple sine model, 2) Ta climatology, 3) PET climatologoy, 4) Ta-variation, 5) PET variation
%lambda_flag 0) optimized, 1) PERSIANN, 2) 0.60 
%NUMEN - number of ensembles used in EnKF or EnKS analysis...not used if filter_flag  = 1 or 3
%Q_fixed - if = 999 than whiten tune, otherwise it sets Q
%P_inflation
%logn_var...variance of multiplicative ensemble perturbations...not sued if filter_flag = 1 or 3....setting to zero means all rainfall error is additive
%slope parameter API - not used if API_model_flag = 0
%location flag 0) CONUS, 1) AMMA, 2) Global 3) Australia 31 4) Australia 240 5) Australia, 0.25-degree continental
%window size - time scale at which rainfall correction is applied 3 to 10 days is recommended