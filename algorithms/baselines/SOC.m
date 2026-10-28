function [X_out,F_val,sparsity,cpu_time,iter,succ_flag,history] = SOC(L,option)
%SOC Splitting orthogonality constraints method for sparse PCA.
% SPDX-License-Identifier: MIT
% Adapted from https://github.com/chenshixiang/ManPG (MIT License).
% This version uses a unified interface, diagnostics, and stopping modes.
% verbose=0 prints only the final line; verbose=1 prints every iteration.

start_clock = tic;
P = option.phi_init;
gamma = option.gamma;
if isfield(option,'linear_term') && ~isempty(option.linear_term)
    linear_term = option.linear_term;
else
    linear_term = zeros(size(P));
end
maxiter = option.maxiter;
tol = option.tol;
zero_tol = option.zero_tol;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end

[n,p] = size(P);
Ip = eye(p);
if isfield(option,'penalty_factor') && ~isempty(option.penalty_factor)
    penalty_factor = option.penalty_factor;
else
    penalty_factor = 1;
end
L_norm = abs(eigs(L,1,'largestabs'));
rho = penalty_factor*L_norm;
system_solver = decomposition(2*rho*eye(n)-L,'chol');
Q = P;
X_old = P;
Z = zeros(n,p);
b = Z;
split_residual = inf;
step_diff_X = inf;
orthogonality = norm(P'*P-Ip,'fro');
F_current = NaN;
succ_flag = 0;
stop_reason = "maxiter";

for iter = 1:maxiter
    P_previous = P;
    X = system_solver\(rho*(P-Z)+rho*(Q-b)-linear_term);
    step_diff_X = norm(X-X_old,'fro');
    X_old = X;
    Q = sign(X+b).*max(0,abs(X+b)-gamma/rho);
    [U,~,V] = svd(X+Z,'econ');
    P = U*V';
    iterate_change = norm(P-P_previous,'fro');
    Z = Z+X-P;
    b = b+X-Q;
    normX = norm(X,'fro');
    split_residual = norm(X-Q,'fro')/max([1,normX,norm(Q,'fro')])+norm(X-P,'fro')/max([1,normX,sqrt(p)]);

    if comparison_mode || verbose==1
        LP = L*P;
        F_current = -0.5*sum(P.*LP,'all')+sum(linear_term.*P,'all')+gamma*sum(abs(P),'all');
        orthogonality = norm(P'*P-Ip,'fro');
    end
    if verbose==1
        fprintf('[SOC %6d] F=% .12e split=%.3e orth=%.3e step=%.3e split_step=%.3e\n',iter,F_current,split_residual,orthogonality,iterate_change,step_diff_X);
    end

    if iter>2
        if comparison_mode==2
            converged = split_residual<=tol && iterate_change<=tol && orthogonality<=tol;
            current_reason = "step_feasibility_and_split_tol";
        elseif comparison_mode
            converged = split_residual<=tol && iterate_change<=tol && F_current-F_reference<=tol && orthogonality<=tol;
            current_reason = "objective_step_feasibility_and_split_tol";
        else
            converged = split_residual<=tol && step_diff_X<=tol;
            current_reason = "split_and_step_tol";
        end
        if converged
            succ_flag = 1;
            stop_reason = current_reason;
            break
        end
    end
end

cpu_time = toc(start_clock);
X_out = P;
if isnan(F_current)
    LP = L*P;
    F_current = -0.5*sum(P.*LP,'all')+sum(linear_term.*P,'all')+gamma*sum(abs(P),'all');
    orthogonality = norm(P'*P-Ip,'fro');
end
F_val = F_current;
sparsity = nnz(abs(P)<=zero_tol)/numel(P);
history.split_residual = split_residual;
history.step_diff_X = step_diff_X;
history.iterate_change = iterate_change;
history.orthogonality = orthogonality;
history.stop_reason = stop_reason;
history.L_norm = L_norm;
fprintf('SOC final: iter=%d flag=%d F=% .12e split=%.3e orth=%.3e time=%.5f s\n',iter,succ_flag,F_val,split_residual,orthogonality,cpu_time);
end
