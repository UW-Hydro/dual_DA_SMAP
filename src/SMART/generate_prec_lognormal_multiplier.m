function [mult_factor] = generate_prec_lognormal_multiplier(logn_var, phi, n)
% Generate a time series of lognormal multipliers
% logn_var: variance of the multiplier
% phi: autocorrelation coefficient of the underlying AR(1) process
% n: length of time series

% Calculate mu and sigma (for the underlying normal dist)
mu = -0.5 * log(logn_var + 1);
sigma = sqrt(log(logn_var + 1));

% Calculate std of white noise Z and generate Z
scale = sigma * sqrt(1 - phi^2);
Z = randn(1, n) * scale;

% AR(1)
% Initialize
ar1(1:n) = 0;
ar1(1) = Z(1);
% Loop over each time step
for t=2:n
    ar1(t) = mu + phi * (ar1(t-1) - mu) + Z(t);
end

% Calculate final lognormal multiplier
mult_factor = exp(ar1);


