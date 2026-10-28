function [X_out,F_val,sparsity,cpu_time,iter,succ_flag,history] = RieSub(L,option)
%RIESUB Riemannian subgradient method for sparse PCA.
% SPDX-License-Identifier: MIT
% Adapted from Re_sub_grad_spca.m in the official ManPG MATLAB package:
% https://github.com/chenshixiang/ManPG (MIT License).
% This version adds selectable stepsizes, a unified interface, and diagnostics.
% The standard mode is a fixed-horizon method and returns flag=0.
% verbose=0 prints only the final line; verbose=1 prints every iteration.

start_clock = tic;
X = option.phi_init;
gamma = option.gamma;
maxiter = option.maxiter;
tol = option.tol;
eta0 = option.eta0;
zero_tol = option.zero_tol;
stepsize_rule = option.stepsize_rule;
stepsize_beta = option.stepsize_beta;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end

[~,p] = size(X);
Ip = eye(p);
LX = L*X;
F_val = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
feasibility = norm(X'*X-Ip,'fro');
iterate_change = inf;
succ_flag = 0;
stop_reason = "fixed_horizon";
iter = 0;

for k = 1:maxiter
    X_previous = X;
    negative_euclidean_subgradient = LX-gamma*sign(X);
    XG = X'*negative_euclidean_subgradient;
    riemannian_direction = negative_euclidean_subgradient-0.5*X*(XG+XG');
    eta = compute_stepsize_riesub(eta0,stepsize_rule,stepsize_beta,k);
    X_trial = X+eta*riemannian_direction;
    trial_Gram = X_trial'*X_trial;
    trial_Gram = 0.5*(trial_Gram+trial_Gram');
    [basis,singular_value_matrix,right_basis] = svd(trial_Gram);
    singular_values = max(diag(singular_value_matrix),eps(class(X)));
    X = X_trial*(basis*diag(1./sqrt(singular_values))*right_basis');
    LX = L*X;
    iter = k;
    iterate_change = norm(X-X_previous,'fro');
    feasibility = norm(X'*X-Ip,'fro');
    F_val = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');

    if verbose==1
        fprintf('[RieSub %6d] F=% .12e orth=%.3e step=%.3e eta=%.3e\n',iter,F_val,feasibility,iterate_change,eta);
    end
    if comparison_mode==2 && iterate_change<=tol && feasibility<=tol
        succ_flag = 1;
        stop_reason = "step_and_feasibility_tol";
        break
    elseif comparison_mode==1 && F_val-F_reference<=tol && iterate_change<=tol && feasibility<=tol
        succ_flag = 1;
        stop_reason = "objective_step_and_feasibility_tol";
        break
    end
end

cpu_time = toc(start_clock);
X_out = X;
sparsity = nnz(abs(X)<=zero_tol)/numel(X);
negative_euclidean_subgradient = LX-gamma*sign(X);
XG = X'*negative_euclidean_subgradient;
riemannian_direction = negative_euclidean_subgradient-0.5*X*(XG+XG');
history.stationarity = norm(riemannian_direction,'fro');
history.iterate_change = iterate_change;
history.feasibility = feasibility;
history.stop_reason = stop_reason;
history.stepsize_rule = stepsize_rule;
history.stepsize_beta = stepsize_beta;
history.eta0 = eta0;
fprintf('RieSub final: iter=%d flag=%d F=% .12e orth=%.3e step=%.3e time=%.5f s\n',iter,succ_flag,F_val,feasibility,iterate_change,cpu_time);
end

function eta = compute_stepsize_riesub(eta0,stepsize_rule,stepsize_beta,k)
if strcmp(stepsize_rule,'power_half')
    eta = eta0/k^0.5;
elseif strcmp(stepsize_rule,'power_three_quarters')
    eta = eta0/k^0.75;
else
    eta = eta0*stepsize_beta^(k-1);
end
end
