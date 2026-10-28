function [L,X_init,A,components] = Gen_SPCA_Synthetic(n,p,m,seed)
%GEN_SPCA_SYNTHETIC Five profiles with Gaussian noise of variance 0.25.
% Profiles follow Huang and Wei (2022, Figure 4). Columns of A are centered
% and normalized to unit norm; L = 2*A'*A. X_init is a Gaussian QR point.
% n: features; p: components; m: samples.

if ~isscalar(n) || n < 2 || n ~= floor(n)
    error('n must be an integer greater than or equal to 2.');
end

if ~isscalar(p) || p < 1 || p ~= floor(p) || p > n
    error('p must be an integer satisfying 1 <= p <= n.');
end

if ~isscalar(m) || m < 5 || m ~= floor(m)
    error('m must be an integer greater than or equal to 5.');
end

if ~isscalar(seed) || ~isfinite(seed) || seed < 0 || seed ~= floor(seed)
    error('seed must be a nonnegative integer.');
end

rng(seed,'twister');

t = linspace(0,1,n);

pc1 = 0.5*double(t>=0.6 & t<=0.8);
pc2 = 0.5*double(t>=0.2 & t<=0.4);
pc3 = 0.8*exp(-(t-0.5).^2/5e-3);
pc4 = 0.4*exp(-(t-0.15).^2/1e-3)+ 0.4*exp(-(t-0.85).^2/1e-3);
pc5 = 0.4*exp(-(t-0.05).^2/1e-3)- 0.4*exp(-(t-0.95).^2/1e-3);

components = [pc1;pc2;pc3;pc4;pc5];

% Distribute the observations as evenly as possible among the five
% components. When m is divisible by five, every component is repeated
% exactly m/5 times, as in the paper.
component_counts = floor(m/5)*ones(1,5);
component_counts(1:mod(m,5)) = component_counts(1:mod(m,5))+1;

A = zeros(m,n);
first_row = 1;
noise_std = sqrt(0.25);

for component_index = 1:5
    number_of_rows = component_counts(component_index);
    row_indices = first_row:first_row+number_of_rows-1;
    A(row_indices,:) = repmat(components(component_index,:),number_of_rows,1)+ noise_std*randn(number_of_rows,n);
    first_row = first_row+number_of_rows;
end

% Center every column.
A = A - mean(A,1);

% Normalize every column to unit Euclidean norm.
column_norms = vecnorm(A,2,1);
column_norms(column_norms == 0) = 1;
A = A ./ column_norms;

% Match -trace(X'*A'*A*X).
L = 2*(A'*A);
L = 0.5*(L+L');

% Use a random Gaussian Stiefel point as the common initial point.
G = randn(n,p);
[X_init,~] = qr(G,0);
end
