function [X_out,F_val,orth_err,sparsity,cpu_time,iter,succ_flag,history] = LSALM(L,option)
%LSALM Linearized smoothing augmented Lagrangian method for sparse PCA.
% SPDX-License-Identifier: MIT
% Independent implementation based on Algorithm 1 of arXiv:2604.04130.
% This is not an official implementation supplied by the paper's authors.
% verbose=0 prints only the final line; verbose=1 prints every iteration.

start_clock = tic;
X = option.phi_init;
gamma = option.gamma;
if isfield(option,'linear_term') && ~isempty(option.linear_term)
    linear_term = option.linear_term;
else
    linear_term = zeros(size(X));
end
maxiter = option.maxiter;
tol = option.tol;
beta = option.beta;
dual_epsilon = option.dual_epsilon;
zero_tol = option.zero_tol;
RX = option.RX;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end

L_norm = abs(eigs(L,1,'largestabs'));
lambda = option.lambda_scale/L_norm;
if option.scale_with_L
    rho = option.rho*L_norm;
    r = option.r*L_norm;
    alpha = option.alpha*L_norm;
    RY = option.RY*L_norm;
else
    rho = option.rho;
    r = option.r;
    alpha = option.alpha;
    RY = option.RY;
end

p = size(X,2);
Ip = eye(p);
Y = zeros(p);
Z = X;
LX = L*X;
G = X'*X-Ip;
G = 0.5*(G+G');
inverse_lambda = 1/lambda;
denominator = r+inverse_lambda;
threshold = gamma/denominator;
primal_change = inf;
F_current = NaN;
succ_flag = 0;
stop_reason = "maxiter";

for iter = 1:maxiter
    augmented_gradient = linear_term-LX+2*X*(Y+rho*G);
    proximal_center = (inverse_lambda*X+r*Z-augmented_gradient)/denominator;
    X_next = sign(proximal_center).*min(max(abs(proximal_center)-threshold,0),RX);
    Z_next = Z+beta*(X_next-Z);
    G_next = X_next'*X_next-Ip;
    G_next = 0.5*(G_next+G_next');
    Y_next = Y+alpha*(G_next-dual_epsilon*Y);
    Y_next = 0.5*(Y_next+Y_next');
    dual_norm = norm(Y_next,'fro');
    if dual_norm>RY
        Y_next = (RY/dual_norm)*Y_next;
    end
    LX_next = L*X_next;
    primal_change = norm(X_next-X,'fro');
    feasibility = norm(G_next,'fro');
    X = X_next;
    Y = Y_next;
    Z = Z_next;
    LX = LX_next;
    G = G_next;

    if comparison_mode || verbose==1
        F_current = -0.5*sum(X.*LX,'all')+sum(linear_term.*X,'all')+gamma*sum(abs(X),'all');
    end
    if verbose==1
        fprintf('[LSALM %6d] F=% .12e orth=%.3e step=%.3e dual=%.3e\n',iter,F_current,feasibility,primal_change,dual_norm);
    end

    if comparison_mode==2
        converged = primal_change<=tol && feasibility<=tol;
        current_reason = "step_and_feasibility_tol";
    elseif comparison_mode
        converged = F_current-F_reference<=tol && primal_change<=tol && feasibility<=tol;
        current_reason = "objective_step_and_feasibility_tol";
    else
        converged = primal_change<=tol && feasibility<=tol;
        current_reason = "paper_step_and_feasibility_tol";
    end
    if converged
        succ_flag = 1;
        stop_reason = current_reason;
        break
    end
end

cpu_time = toc(start_clock);
X_out = X;
if isnan(F_current)
    F_current = -0.5*sum(X.*LX,'all')+sum(linear_term.*X,'all')+gamma*sum(abs(X),'all');
end
F_val = F_current;
orth_err = norm(G,'fro');
sparsity = nnz(abs(X)<=zero_tol)/numel(X);
history.primal_change = primal_change;
history.stop_reason = stop_reason;
history.L_norm = L_norm;
history.lambda = lambda;
history.rho = rho;
history.r = r;
history.alpha = alpha;
history.RY = RY;
history.RX = RX;
fprintf('LSALM final: iter=%d flag=%d F=% .12e orth=%.3e step=%.3e time=%.5f s\n',iter,succ_flag,F_val,orth_err,primal_change,cpu_time);
end
