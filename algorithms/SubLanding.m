function [X_out,F_val,orth_err,sparsity,cpu_time,iter,succ_flag,history] = SubLanding(L,option)
% SPDX-License-Identifier: MIT
%SUBLANDING SubLanding with safe-region backtracking.
% verbose=0 prints only the final line; verbose=1 prints every iteration.

start_clock = tic;
X = option.phi_init;
[~,p] = size(X);
gamma = option.gamma;
maxiter = option.maxiter;
tol = option.tol;
eta0 = option.eta0;
lambda = option.lambda;
alpha = option.alpha;
zero_tol = option.zero_tol;
stepsize_rule = option.stepsize_rule;
stepsize_beta = option.stepsize_beta;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end
epsilon = get_option_sublanding(option,'epsilon',0.5);
safe_backtrack_tau = get_option_sublanding(option,'safe_backtrack_tau',0.5);
safe_max_backtracks = get_option_sublanding(option,'safe_max_backtracks',60);
eta_min = get_option_sublanding(option,'eta_min',1e-8);

Ip = eye(p);
LX = L*X;
XtX = X'*X;
E = XtX-Ip;
feasibility = norm(E,'fro');
F_current = objective_sublanding(X,LX,gamma);
subgradient = -LX+gamma*sign(X);
AtX = subgradient'*X;
tangent_gradient = subgradient*XtX-X*AtX;
tangent_norm = norm(tangent_gradient,'fro');
iterate_change = inf;
succ_flag = 0;
stop_reason = "maxiter";
total_backtracks = 0;
backtracked_iterations = 0;
maximum_backtracks_per_iteration = 0;
iter = 0;

for k = 1:maxiter
    eta = compute_stepsize_sublanding(eta0,stepsize_rule,stepsize_beta,k);
    normal_base = X*E;
    backtracks = 0;
    accepted = false;

    while backtracks<=safe_max_backtracks
        if alpha==0
            zeta = lambda;
        else
            zeta = lambda*eta^alpha;
        end
        step = -eta*tangent_gradient-zeta*normal_base;
        X_trial = X+step;
        trial_XtX = X_trial'*X_trial;
        trial_XtX = 0.5*(trial_XtX+trial_XtX');
        trial_E = trial_XtX-Ip;
        trial_feasibility = norm(trial_E,'fro');
        if trial_feasibility<=epsilon
            accepted = true;
            break
        end
        eta = safe_backtrack_tau*eta;
        backtracks = backtracks+1;
        if eta<eta_min
            break
        end
    end

    total_backtracks = total_backtracks+backtracks;
    if backtracks>0
        backtracked_iterations = backtracked_iterations+1;
    end
    maximum_backtracks_per_iteration = max(maximum_backtracks_per_iteration,backtracks);
    if ~accepted
        succ_flag = 2;
        iter = k-1;
        stop_reason = "safe_region_backtracking_failed";
        break
    end

    X = X_trial;
    LX = L*X;
    XtX = trial_XtX;
    E = trial_E;
    feasibility = trial_feasibility;
    F_current = objective_sublanding(X,LX,gamma);
    subgradient = -LX+gamma*sign(X);
    AtX = subgradient'*X;
    tangent_gradient = subgradient*XtX-X*AtX;
    tangent_norm = norm(tangent_gradient,'fro');
    iterate_change = norm(step,'fro');
    iter = k;

    if verbose==1
        fprintf('[SubLanding %6d] F=% .12e tangent=%.3e orth=%.3e step=%.3e eta=%.3e backtracks=%d\n',iter,F_current,tangent_norm,feasibility,iterate_change,eta,backtracks);
    end
    if comparison_mode==2
        converged = iterate_change<=tol && feasibility<=tol;
        current_reason = "step_and_feasibility_tol";
    elseif comparison_mode==1
        converged = F_current-F_reference<=tol && iterate_change<=tol && feasibility<=tol;
        current_reason = "objective_step_and_feasibility_tol";
    else
        converged = tangent_norm<=tol && feasibility<=tol;
        current_reason = "stationarity_and_feasibility_tol";
    end
    if converged
        succ_flag = 1;
        stop_reason = current_reason;
        break
    end
end

cpu_time = toc(start_clock);
X_out = X;
F_val = F_current;
orth_err = feasibility;
sparsity = nnz(abs(X)<=zero_tol)/numel(X);
history.stationarity = tangent_norm;
history.iterate_change = iterate_change;
history.total_backtracks = total_backtracks;
history.backtracked_iterations = backtracked_iterations;
history.max_backtracks_per_iteration = maximum_backtracks_per_iteration;
history.stop_reason = stop_reason;
history.stepsize_rule = stepsize_rule;
history.stepsize_beta = stepsize_beta;
history.eta0 = eta0;
fprintf('SubLanding final: iter=%d flag=%d F=% .12e tangent=%.3e orth=%.3e time=%.5f s\n',iter,succ_flag,F_val,tangent_norm,orth_err,cpu_time);
end

function eta = compute_stepsize_sublanding(eta0,stepsize_rule,stepsize_beta,k)
if strcmp(stepsize_rule,'power_half')
    eta = eta0/k^0.5;
elseif strcmp(stepsize_rule,'power_three_quarters')
    eta = eta0/k^0.75;
else
    eta = eta0*stepsize_beta^(k-1);
end
end

function value = objective_sublanding(X,LX,gamma)
value = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
end

function value = get_option_sublanding(option,name,default_value)
if isfield(option,name) && ~isempty(option.(name))
    value = option.(name);
else
    value = default_value;
end
end
