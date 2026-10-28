function [X_out,F_val,orth_err,sparsity,cpu_time,iter,succ_flag,history] = RADMM(L,option)
%RADMM Independent MATLAB implementation of Riemannian ADMM for sparse PCA.
% SPDX-License-Identifier: MIT
% The implementation follows the update equations in the RADMM paper and
% uses a project-specific interface, local kernels, and stopping criteria.
% This is not an official implementation supplied by the paper's authors.
%
% verbose=0 prints only the final line; verbose=1 prints every iteration.

start_clock = tic;
X = option.phi_init;
gamma = option.gamma;
maxiter = option.maxiter;
tol = option.tol;
zero_tol = option.zero_tol;
verbose = option.verbose;
comparison_mode = option.comparison_mode;

if isfield(option,'linear_term') && ~isempty(option.linear_term)
    linear_term = option.linear_term;
else
    linear_term = zeros(size(X));
end
if comparison_mode==1
    reference_objective = option.reference_objective;
end

[n,p] = size(X);
identity_p = eye(p,class(X));
L_norm = abs(eigs(L,1,'largestabs'));
penalty = L_norm;
primal_stepsize = 1/(2*penalty);
smoothing = 1e-12;

Z = X;
dual_variable = zeros(n,p,class(X));
LX = L*X;
F_val = objective_value(X,LX,linear_term,gamma);
orth_err = norm(X'*X-identity_p,'fro');
step_norm = inf;
split_residual = inf;
objective_change = inf;
succ_flag = 0;
stop_reason = "maxiter";

for iter = 1:maxiter
    X_previous = X;
    previous_objective = F_val;

    augmented_gradient = linear_term-LX+dual_variable+penalty*(X-Z);
    tangent_gradient = tangent_projection(X,augmented_gradient);
    X = polar_retraction(X-primal_stepsize*tangent_gradient);

    proximal_center = X+dual_variable/penalty;
    threshold = gamma*(1+penalty*smoothing)/penalty;
    sparse_response = soft_threshold(proximal_center,threshold);
    Z = (sparse_response+smoothing*dual_variable+smoothing*penalty*X)/(1+smoothing*penalty);
    dual_variable = dual_variable+penalty*(X-Z);

    LX = L*X;
    F_val = objective_value(X,LX,linear_term,gamma);
    step_norm = norm(X-X_previous,'fro');
    split_residual = norm(X-Z,'fro')/max([1,norm(X,'fro'),norm(Z,'fro')]);
    objective_change = abs(F_val-previous_objective);
    orth_err = norm(X'*X-identity_p,'fro');

    if verbose==1
        fprintf('[RADMM %6d] F=% .12e split=%.3e orth=%.3e step=%.3e dF=%.3e\n',iter,F_val,split_residual,orth_err,step_norm,objective_change);
    end

    if iter>2
        if comparison_mode==2
            converged = step_norm<=tol && orth_err<=tol && split_residual<=tol;
            current_reason = "step_feasibility_and_split_tol";
        elseif comparison_mode==1
            converged = F_val-reference_objective<=tol && step_norm<=tol && orth_err<=tol && split_residual<=tol;
            current_reason = "objective_step_feasibility_and_split_tol";
        else
            converged = split_residual<tol && objective_change<=1e-8;
            current_reason = "paper_split_and_objective_change_tol";
        end
        if converged
            succ_flag = 1;
            stop_reason = current_reason;
            break
        end
    end
end

cpu_time = toc(start_clock);
X_out = X;
sparsity = nnz(abs(X)<=zero_tol)/numel(X);
history.step_norm = step_norm;
history.relative_split_residual = split_residual;
history.objective_change = objective_change;
history.stop_reason = stop_reason;
history.L_norm = L_norm;
fprintf('RADMM final: iter=%d flag=%d F=% .12e split=%.3e orth=%.3e time=%.5f s\n',iter,succ_flag,F_val,split_residual,orth_err,cpu_time);
end

function projected = tangent_projection(X,G)
cross_term = X'*G;
projected = G-0.5*X*(cross_term+cross_term');
end

function X_new = polar_retraction(Y)
gram_matrix = Y'*Y;
gram_matrix = 0.5*(gram_matrix+gram_matrix');
[basis,singular_value_matrix,right_basis] = svd(gram_matrix);
singular_values = max(diag(singular_value_matrix),eps(class(Y)));
X_new = Y*(basis*diag(1./sqrt(singular_values))*right_basis');
end

function Z = soft_threshold(Y,threshold)
Z = sign(Y).*max(abs(Y)-threshold,0);
end

function value = objective_value(X,LX,linear_term,gamma)
value = -0.5*sum(X.*LX,'all')+sum(linear_term.*X,'all')+gamma*sum(abs(X),'all');
end
