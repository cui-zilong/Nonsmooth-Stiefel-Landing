clear;
clc;

% Quick installation and functionality check.
% This script is not used to generate any numerical result in the paper.

demo_directory = fileparts(mfilename('fullpath'));
repository_root = fileparts(demo_directory);
addpath(repository_root,'-begin');
setup_paths;

rng(0,'twister');
n = 100;
m = 50;
p = 3;
gamma = 0.5;
[L,X_initial] = Gen_SPCA_Random(n,p,m,0);

warm_option = struct('phi_init',X_initial,'gamma',gamma,'maxiter',50,'tol',1e-3,'eta0',1,'zero_tol',0,'stepsize_rule','power_three_quarters','stepsize_beta',0.999,'verbose',0,'comparison_mode',false);
X_initial = RieSub(L,warm_option);

option = struct();
option.phi_init = X_initial;
option.gamma = gamma;
option.maxiter = 3000;
option.tol = 1e-3;
option.zero_tol = 1e-5;
option.epsilon = 0.5;
option.lambda = 5/12;
option.dual_theta = 1.95;
option.initial_primal_stepsize = 1.20;
option.backtrack_factor = 0.4;
option.nonmonotone_memory = 0.95;
option.merit_rho = 1e-3;
option.merit_sigma_factor = 1.1;
option.merit_decrease_fraction = 1e-3;
option.verbose = 0;
option.comparison_mode = false;

[X,F,orthogonality,sparsity,cpu_time,iterations,flag,history] = IPGLanding(L,option);
assert(all(isfinite(X(:))),'IPGLanding returned NaN or Inf.');
assert(isfinite(F) && isfinite(orthogonality),'The reported diagnostics are not finite.');
assert(orthogonality<1e-4,'The orthogonality residual is unexpectedly large.');
assert(flag==1 && history.stationarity<=option.tol,'IPGLanding did not meet the stopping tolerance.');

Result = table(F,sparsity,iterations,cpu_time,history.stationarity,orthogonality,flag,'VariableNames',{'Objective','Sparsity','Iterations','TimeSeconds','Stationarity','Orthogonality','Flag'});
disp(Result);
fprintf('Quick-start check passed.\n');
