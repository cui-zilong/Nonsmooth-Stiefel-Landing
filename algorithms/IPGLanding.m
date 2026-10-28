function [X_out,F_val,orth_err,sparsity,cpu_time,iter,succ_flag,history] = IPGLanding(L,option)
%IPGLANDING Inexact proximal landing method for sparse PCA.
% SPDX-License-Identifier: MIT
%
%   min  -0.5*trace(X'*L*X)+gamma*||X||_1
%   s.t. X'*X=I_p.
%
% The spectral norm of L is computed once inside this function and the
% proximal parameter is fixed at t=2/||L||_2. The reported iteration count
% is the number of accepted X updates. No iterate history is stored.

start_clock = tic;
X = option.phi_init;
gamma = option.gamma;
maxiter = option.maxiter;
tol = option.tol;
epsilon = option.epsilon;
lambda = option.lambda;
dual_theta = option.dual_theta;
memory = option.nonmonotone_memory;
rho = option.merit_rho;
sigma_factor = option.merit_sigma_factor;
decrease_fraction = option.merit_decrease_fraction;
eta0 = option.initial_primal_stepsize;
backtrack_factor = option.backtrack_factor;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end

[n,p] = size(X);
Ip = eye(p,class(X));
L_norm = abs(eigs(L,1,'largestabs'));
t = 2/L_norm;
inverse_t = 1/t;
threshold = t*gamma;
dual_step = dual_theta/(4*t*(1+epsilon));
normal_certificate_threshold = lambda*(1-epsilon);

LX = L*X;
E = X'*X-Ip;
orth_err = norm(E,'fro');
sum_abs_X = sum(abs(X),'all');
F_val = -0.5*sum(X.*LX,'all')+gamma*sum_abs_X;
L_F = L_norm*sqrt(p*(1+epsilon))+gamma*sqrt(n*p);
dual_radius = L_F/sqrt(1-epsilon);
dual_radius_squared = dual_radius^2;
sigma = sigma_factor*(rho+lambda*L_F*sqrt(1+epsilon))/(2*lambda*(1-epsilon));

Gamma = zeros(p,class(X));
reference_value = F_val+sigma*orth_err;
reference_weight = 1;
iterate_change = inf;
stationarity = inf;
response_evaluations = 0;
accepted_updates = 0;
total_multiplier_updates = 0;
total_backtracks = 0;
succ_flag = 0;
stop_reason = "maximum_X_updates";
terminate = false;

while accepted_updates<=maxiter && ~terminate
    accepted = false;
    while ~accepted && ~terminate
        response_evaluations = response_evaluations+1;
        Y = X*(Ip+2*t*Gamma)+t*LX;
        Y = sign(Y).*max(abs(Y)-threshold,0);
        V = Y-X;
        v_norm = norm(V,'fro');
        stationarity = inverse_t*v_norm;

        if verbose==1
            fprintf('[IPGLanding %6d] Xiter=%d F=% .12e stat=%.3e orth=%.3e step=%.3e\n',response_evaluations,accepted_updates,F_val,stationarity,orth_err,iterate_change);
        end

        if comparison_mode==2
            converged = iterate_change<=tol && orth_err<=tol;
            current_reason = "step_and_feasibility_tol";
        elseif comparison_mode==1
            converged = F_val-F_reference<=tol && iterate_change<=tol && orth_err<=tol;
            current_reason = "objective_step_and_feasibility_tol";
        else
            converged = max(stationarity,orth_err)<=tol;
            current_reason = "stationarity_and_feasibility_tol";
        end
        if converged
            succ_flag = 1;
            stop_reason = current_reason;
            terminate = true;
            break
        end
        if accepted_updates==maxiter
            terminate = true;
            break
        end

        XV = X'*V;
        R = XV+XV';
        Gamma_map = Gamma-dual_step*R;
        direction = V-X*(0.5*R+lambda*E);
        measure = rho*(inverse_t*v_norm^2+orth_err);
        smooth_linear_change = -sum(direction.*LX,'all');
        Ldirection = L*direction;
        smooth_quadratic_change = -0.5*sum(direction.*Ldirection,'all');
        direction_gram = direction'*direction;
        gram_cross_term = X'*direction+direction'*X;

        eta = eta0;
        backtracks_this_response = 0;
        E_trial = E+eta*gram_cross_term+eta^2*direction_gram;
        orth_trial = norm(E_trial,'fro');
        if orth_trial<=epsilon
            X_trial = X+eta*direction;
            sum_abs_trial = sum(abs(X_trial),'all');
            F_trial = F_val+eta*smooth_linear_change+eta^2*smooth_quadratic_change+gamma*(sum_abs_trial-sum_abs_X);
            Phi_trial = F_trial+sigma*orth_trial;
            accepted = Phi_trial<=reference_value-decrease_fraction*eta*measure;
        end

        if ~accepted
            multiplier_residual = norm(R,'fro');
            if multiplier_residual<=normal_certificate_threshold
                X_unit = X+direction;
                sum_abs_unit = sum(abs(X_unit),'all');
                ell = smooth_linear_change+gamma*(sum_abs_unit-sum_abs_X)+sigma*orth_err*(-2*lambda*(1-epsilon)+multiplier_residual);
                if ell<=-measure
                    while ~accepted
                        eta = backtrack_factor*eta;
                        backtracks_this_response = backtracks_this_response+1;
                        E_trial = E+eta*gram_cross_term+eta^2*direction_gram;
                        orth_trial = norm(E_trial,'fro');
                        if orth_trial<=epsilon
                            X_trial = X+eta*direction;
                            sum_abs_trial = sum(abs(X_trial),'all');
                            F_trial = F_val+eta*smooth_linear_change+eta^2*smooth_quadratic_change+gamma*(sum_abs_trial-sum_abs_X);
                            Phi_trial = F_trial+sigma*orth_trial;
                            accepted = Phi_trial<=reference_value-decrease_fraction*eta*measure;
                        end
                    end
                end
            end
        end

        if accepted
            multiplier_norm_squared = sum(Gamma_map.*Gamma_map,'all');
            if multiplier_norm_squared>dual_radius_squared
                Gamma = (dual_radius/sqrt(multiplier_norm_squared))*Gamma_map;
            else
                Gamma = Gamma_map;
            end
            total_backtracks = total_backtracks+backtracks_this_response;
            new_reference_weight = memory*reference_weight+1;
            reference_value = (memory*reference_weight*reference_value+Phi_trial)/new_reference_weight;
            reference_weight = new_reference_weight;
            X_old = X;
            X = X_trial;
            LX = LX+eta*Ldirection;
            E = E_trial;
            orth_err = orth_trial;
            F_val = F_trial;
            sum_abs_X = sum_abs_trial;
            iterate_change = norm(X-X_old,'fro');
            accepted_updates = accepted_updates+1;
        else
            Gamma = Gamma_map;
            total_multiplier_updates = total_multiplier_updates+1;
        end
    end
end

% Report residuals of the returned matrix rather than the Gram recurrence.
LX_final = L*X;
F_val = -0.5*sum(X.*LX_final,'all')+gamma*sum(abs(X),'all');
orth_err = norm(X'*X-Ip,'fro');
cpu_time = toc(start_clock);
X_out = X;
sparsity = nnz(abs(X)<=option.zero_tol)/numel(X);
iter = accepted_updates;
history.stationarity = stationarity;
history.iterate_change = iterate_change;
history.accepted_updates = accepted_updates;
history.response_evaluations = response_evaluations;
history.total_multiplier_updates = total_multiplier_updates;
history.total_backtracks = total_backtracks;
history.stop_reason = stop_reason;
history.L_norm = L_norm;
history.t = t;
fprintf('IPGLanding final: Xiter=%d responses=%d flag=%d F=% .12e stat=%.3e orth=%.3e time=%.5f s\n',iter,response_evaluations,succ_flag,F_val,stationarity,orth_err,cpu_time);
end
