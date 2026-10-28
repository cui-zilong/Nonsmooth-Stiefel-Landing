function [L,X_init] = Gen_SPCA_Random(n,p,m,seed)
%GEN_SPCA_RANDOM Gaussian data with centered, unit-norm columns.
% n: features; p: components; m: samples. L = 2*A'*A.

rng(seed,'twister');
A = randn(m,n);
A = A-mean(A,1);
column_norms = vecnorm(A,2,1);
column_norms(column_norms==0) = 1;
A = A./column_norms;
L = 2*(A'*A);
[X_init,~] = svd(randn(n,p),0);
end
