function [X_out,F_val,orth_err,sparsity,cpu_time,iter,succ_flag,history] = SLPG(L,option)
%SLPG Sparse-PCA specialization of the nonsmooth SLPG method.
% SPDX-License-Identifier: LGPL-3.0-or-later
% MATLAB adaptation based on PySTOP 0.2.2.
% PySTOP authors: Nachuan Xiao, Lei Wang, Bin Gao, Xin Liu, Ya-xiang Yuan.
% Original package: https://pypi.org/project/pystop/0.2.2/
% Modified 2026-10-26: MATLAB/SPCA interface, stopping rules, BB cap,
% and optional diagnostics. See THIRD_PARTY_NOTICE.md for license details.
% This file is excluded from the repository-level MIT License.
% In comparison mode, trajectory storage can be disabled independently of
% the final diagnostic summary returned in history.

X = option.phi_init;
gamma = option.gamma;
if isfield(option,'linear_term') && ~isempty(option.linear_term)
    linear_term = option.linear_term;
else
    linear_term = zeros(size(X));
end
maxiter = get_option_slpg(option,'maxiter',100);
tol = get_option_slpg(option,'tol',1e-5);
zero_tol = get_option_slpg(option,'zero_tol',1e-5);
verbose = get_option_slpg(option,'verbose',2);
post_process = get_option_slpg(option,'post_process',true);
inner_maxiter = get_option_slpg(option,'inner_maxiter',5);
inner_constant = get_option_slpg(option,'inner_constant',1000);
initial_scale = get_option_slpg(option,'initial_scale',0.01);
initial_steps = get_option_slpg(option,'initial_steps',5);
comparison_mode = get_option_slpg(option,'comparison_mode',false);
record_history = comparison_mode==0 && get_option_slpg(option,'record_history',nargout>=8);
history_stride = get_option_slpg(option,'history_stride',1);
bb_spectral_cap = get_option_slpg(option,'bb_spectral_cap',inf);
if comparison_mode==1
    F_reference = option.reference_objective;
end

start_clock = tic;
[~,p] = size(X);
Ip = eye(p,class(X));
if isfinite(bb_spectral_cap)
    L_norm = abs(eigs(L,1,'largestabs'));
    maximum_stepsize = bb_spectral_cap/L_norm;
else
    L_norm = NaN;
    maximum_stepsize = 1e10;
end
LX = L*X;
smooth_gradient = linear_term-LX;
projected_gradient = stiefel_projected_gradient_slpg(X,smooth_gradient);
L0 = norm(smooth_gradient,'fro')+norm(projected_gradient,'fro');
if ~(isfinite(L0) && L0>0)
    L0 = 1;
end

initial_stepsize = initial_scale/L0;
Lambda = zeros(p,p,class(X));
[Lambda,initial_inner_iterations,initial_inner_residual] = arrow_hurwicz_slpg(X,projected_gradient,initial_stepsize,gamma,Lambda,inner_maxiter,0);
composite_gradient = projected_gradient+X*sym_slpg(Lambda);

F_val = objective_slpg(X,LX,linear_term,gamma);
orth_err = norm(X'*X-Ip,'fro');
iterate_change = inf;
native_residual = inf;
iter = 0;
succ_flag = 0;
stop_reason = "maxiter";
recent_stepsizes = zeros(11,1);
S = [];
Y = [];
bb_fallbacks = 0;
total_inner_iterations = initial_inner_iterations;
current_inner_iterations = 0;
current_inner_residual = initial_inner_residual;

if record_history
    record_capacity = floor(maxiter/history_stride)+2;
    history_iteration = zeros(record_capacity,1);
    history_time = zeros(record_capacity,1);
    history_objective = zeros(record_capacity,1);
    history_feasibility = zeros(record_capacity,1);
    history_native_residual = zeros(record_capacity,1);
    history_iterate_change = zeros(record_capacity,1);
    history_stepsize = zeros(record_capacity,1);
    history_inner_iterations = zeros(record_capacity,1);
    history_inner_residual = zeros(record_capacity,1);
    record_count = 0;
end

for k = 1:maxiter
    if k<=initial_steps
        stepsize = initial_stepsize;
    else
        bb_denominator = sum(S(:).*Y(:));
        bb_numerator = sum(S(:).*S(:));
        if isfinite(bb_denominator) && isfinite(bb_numerator) && abs(bb_denominator)>eps(max(1,abs(bb_numerator)))
            stepsize = min(abs(bb_numerator/bb_denominator),maximum_stepsize);
        else
            stepsize = recent_stepsizes(min(k-1,11));
            bb_fallbacks = bb_fallbacks+1;
        end
    end
    if ~(isfinite(stepsize) && stepsize>0)
        succ_flag = 2;
        stop_reason = "invalid_stepsize";
        break
    end

    X_previous = X;
    composite_gradient_previous = composite_gradient;
    if k>11
        recent_stepsizes(1:10) = recent_stepsizes(2:11);
    end
    recent_stepsizes(min(k,11)) = stepsize;
    proximal_argument = X-stepsize*composite_gradient;
    Y_prox = soft_threshold_slpg(proximal_argument,gamma*stepsize);
    X = normal_correction_slpg(Y_prox);
    S = X-X_previous;

    LX = L*X;
    smooth_gradient = linear_term-LX;
    projected_gradient = stiefel_projected_gradient_slpg(X,smooth_gradient);
    trial_stepsize = mean(recent_stepsizes(1:min(k,11)));
    trial_stepsize = min(max(trial_stepsize,1e-5/L0),min(1e10/L0,maximum_stepsize));
    orth_err = norm(X'*X-Ip,'fro');
    inner_tolerance = inner_constant*orth_err;
    [Lambda,current_inner_iterations,current_inner_residual] = arrow_hurwicz_slpg(X,projected_gradient,trial_stepsize,gamma,Lambda,inner_maxiter,inner_tolerance);
    total_inner_iterations = total_inner_iterations+current_inner_iterations;
    composite_gradient = projected_gradient+X*sym_slpg(Lambda);
    Y = composite_gradient-composite_gradient_previous;

    iterate_change = norm(S,'fro');
    native_residual = iterate_change/stepsize;
    F_val = objective_slpg(X,LX,linear_term,gamma);
    iter = k;

    if record_history && (mod(k,history_stride)==0 || k==1)
        record_count = record_count+1;
        history_iteration(record_count) = k;
        history_time(record_count) = toc(start_clock);
        history_objective(record_count) = F_val;
        history_feasibility(record_count) = orth_err;
        history_native_residual(record_count) = native_residual;
        history_iterate_change(record_count) = iterate_change;
        history_stepsize(record_count) = stepsize;
        history_inner_iterations(record_count) = current_inner_iterations;
        history_inner_residual(record_count) = current_inner_residual;
    end

    if verbose==1 || (verbose==2 && mod(k-1,50)==0)
        fprintf('[SLPG %6d] F=% .12e residual=%.3e orth=%.3e step=%.3e eta=%.3e inner=%d\n',k,F_val,native_residual,orth_err,iterate_change,stepsize,current_inner_iterations);
    end

    if comparison_mode==2
        converged = iterate_change<=tol && orth_err<=tol;
        current_reason = "step_and_feasibility_tol";
    elseif comparison_mode
        converged = F_val-F_reference<=tol && iterate_change<=tol && orth_err<=tol;
        current_reason = "objective_step_and_feasibility_tol";
    else
        converged = native_residual<tol;
        current_reason = "pystop_displacement_residual_tol";
    end
    if converged
        succ_flag = 1;
        stop_reason = current_reason;
        break
    end
end

if record_history && iter>0 && (record_count==0 || history_iteration(record_count)~=iter)
    record_count = record_count+1;
    history_iteration(record_count) = iter;
    history_time(record_count) = toc(start_clock);
    history_objective(record_count) = F_val;
    history_feasibility(record_count) = orth_err;
    history_native_residual(record_count) = native_residual;
    history_iterate_change(record_count) = iterate_change;
    history_stepsize(record_count) = recent_stepsizes(min(iter,11));
    history_inner_iterations(record_count) = current_inner_iterations;
    history_inner_residual(record_count) = current_inner_residual;
end

preprocess_objective = F_val;
preprocess_feasibility = orth_err;
if post_process
    [U,~,V] = svd(X,'econ');
    X = U*V';
    LX = L*X;
    F_val = objective_slpg(X,LX,linear_term,gamma);
    orth_err = norm(X'*X-Ip,'fro');
end

cpu_time = toc(start_clock);
X_out = X;
sparsity = nnz(abs(X)<=zero_tol)/numel(X);

if nargout>=8
    history = struct();
    if record_history
        history.iteration = history_iteration(1:record_count);
        history.time = history_time(1:record_count);
        history.objective = history_objective(1:record_count);
        history.feasibility = history_feasibility(1:record_count);
        history.native_residual_history = history_native_residual(1:record_count);
        history.iterate_change_history = history_iterate_change(1:record_count);
        history.stepsize = history_stepsize(1:record_count);
        history.inner_iterations = history_inner_iterations(1:record_count);
        history.inner_residual = history_inner_residual(1:record_count);
    end
    history.native_residual = native_residual;
    history.iterate_change = iterate_change;
    history.initial_inner_iterations = initial_inner_iterations;
    history.initial_inner_residual = initial_inner_residual;
    history.total_inner_iterations = total_inner_iterations;
    history.total_proximal_evaluations = iter+total_inner_iterations;
    history.bb_fallbacks = bb_fallbacks;
    history.stop_reason = stop_reason;
    history.post_process = logical(post_process);
    history.preprocess_objective = preprocess_objective;
    history.preprocess_feasibility = preprocess_feasibility;
    history.final_objective = F_val;
    history.final_feasibility = orth_err;
    history.pySTOP_version = "0.2.2";
    history.bb_spectral_cap = bb_spectral_cap;
    history.L_norm = L_norm;
end

fprintf('SLPG final: iter=%d flag=%d F=% .12e residual=%.3e orth=%.3e time=%.5f s\n',iter,succ_flag,F_val,native_residual,orth_err,cpu_time);
end

function [Lambda,inner_iterations,inner_residual] = arrow_hurwicz_slpg(X,G,eta,gamma,Lambda,maxiter,tolerance)
Z = X-eta*G;
inner_residual = inf;
inner_iterations = 0;
for j = 1:maxiter
    proximal_argument = Z-eta*(X*sym_slpg(Lambda));
    X_trial = soft_threshold_slpg(proximal_argument,gamma*eta);
    scaled_displacement = (X_trial-X)/eta;
    Lambda_increment = sym_slpg(X'*scaled_displacement);
    Lambda = Lambda+Lambda_increment;
    inner_residual = norm(Lambda_increment,'fro');
    inner_iterations = j;
    if inner_residual<tolerance
        break
    end
end
end

function projected_gradient = stiefel_projected_gradient_slpg(X,G)
projected_gradient = G-X*sym_slpg(X'*G);
end

function X_corrected = normal_correction_slpg(X)
X_corrected = 1.5*X-0.5*X*(X'*X);
end

function value = objective_slpg(X,LX,linear_term,gamma)
value = -0.5*sum(X(:).*LX(:))+sum(linear_term(:).*X(:))+gamma*sum(abs(X(:)));
end

function Y = soft_threshold_slpg(X,threshold)
Y = max(X-threshold,0)+min(X+threshold,0);
end

function S = sym_slpg(A)
S = 0.5*(A+A');
end

function value = get_option_slpg(option,name,default_value)
if isfield(option,name) && ~isempty(option.(name))
    value = option.(name);
else
    value = default_value;
end
end

