function [X_out,F_val,sparsity,cpu_time,iter,succ_flag,num_linesearch,mean_ssn,history] = ManPGAda(L,option)
%MANPGADA Adaptive ManPG method for sparse PCA.
% SPDX-License-Identifier: MIT
% Adapted from https://github.com/chenshixiang/ManPG (MIT License).
% This version uses a unified interface, diagnostics, and stopping modes.
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
inner_iter = option.inner_iter;
zero_tol = option.zero_tol;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end
high_accuracy = isfield(option,'inner_residual_tolerance');

L_norm = abs(eigs(L,1,'largestabs'));
t0 = 1/L_norm;
t = t0;
LX = L*X;
F_val = -0.5*sum(X.*LX,'all')+sum(linear_term.*X,'all')+gamma*sum(abs(X),'all');
p = size(X,2);
Lambda = zeros(p);
[row,column] = find(tril(true(p)));
q = numel(row);
off_diagonal = row~=column;
matrix_rows = [sub2ind([p,p],row,column);sub2ind([p,p],column(off_diagonal),row(off_diagonal))];
matrix_columns = [(1:q)';find(off_diagonal)];
Dn = sparse(matrix_rows,matrix_columns,1,p^2,q);
pDn = (Dn'*Dn)\Dn';
alpha = 1;
line_search_failures = 0;
num_linesearch = 0;
total_ssn_iterations = 0;
outer_sweeps = 0;
iterate_change = inf;
stationarity = inf;
succ_flag = 0;
stop_reason = "maxiter";
iter = 0;

for k = 0:maxiter
    if high_accuracy
        inner_tolerance = option.inner_residual_tolerance^2;
        cg_floor = 0;
    elseif alpha<1e-4 || line_search_failures>10
        inner_tolerance = max(1e-15,min(1e-13,1e-5*tol*t^2));
        cg_floor = 1e-7;
    else
        inner_tolerance = max(1e-13,min(1e-11,1e-3*tol*t^2));
        cg_floor = 1e-7;
    end

    [proximal_point,ssn_iterations,Lambda,~,~] = ManPGSSN(X,t,X+t*(LX-linear_term),gamma*t,inner_tolerance,inner_iter,Lambda,cg_floor,Dn,pDn);
    total_ssn_iterations = total_ssn_iterations+ssn_iterations;
    outer_sweeps = outer_sweeps+1;
    direction = proximal_point-X;
    direction_norm_squared = norm(direction,'fro')^2;
    stationarity = sqrt(direction_norm_squared)/t;
    orthogonality = norm(X'*X-eye(size(X,2)),'fro');

    if verbose==1
        fprintf('[ManPG-Ada %6d] F=% .12e stat=%.3e orth=%.3e step=%.3e t=%.3e ssn=%d\n',iter,F_val,stationarity,orthogonality,iterate_change,t,ssn_iterations);
    end

    if comparison_mode==2
        converged = iterate_change<=tol && orthogonality<=tol;
        current_reason = "step_and_feasibility_tol";
    elseif comparison_mode
        converged = F_val-F_reference<=tol && iterate_change<=tol && orthogonality<=tol;
        current_reason = "objective_step_and_feasibility_tol";
    else
        converged = stationarity<=tol;
        current_reason = "stationarity_tol";
    end
    if converged
        succ_flag = 1;
        stop_reason = current_reason;
        break
    end
    if iter==maxiter
        break
    end

    alpha = 1;
    trial_point = polar_retraction(proximal_point);
    L_trial = L*trial_point;
    F_trial = -0.5*sum(trial_point.*L_trial,'all')+sum(linear_term.*trial_point,'all')+gamma*sum(abs(trial_point),'all');
    backtracked = false;

    while F_trial>=F_val-0.5*alpha*direction_norm_squared/t
        alpha = 0.5*alpha;
        num_linesearch = num_linesearch+1;
        backtracked = true;
        if alpha<1e-4
            line_search_failures = line_search_failures+1;
            break
        end
        proximal_point = X+alpha*direction;
        trial_point = polar_retraction(proximal_point);
        L_trial = L*trial_point;
        F_trial = -0.5*sum(trial_point.*L_trial,'all')+sum(linear_term.*trial_point,'all')+gamma*sum(abs(trial_point),'all');
    end

    X_previous = X;
    X = trial_point;
    iter = iter+1;
    LX = L_trial;
    F_val = F_trial;
    iterate_change = norm(X-X_previous,'fro');
    if backtracked
        t = max(t0,t/1.01);
    else
        t = 1.01*t;
    end
end

cpu_time = toc(start_clock);
X_out = X;
sparsity = nnz(abs(X)<=zero_tol)/numel(X);
mean_ssn = total_ssn_iterations/max(outer_sweeps,1);
orthogonality = norm(X'*X-eye(size(X,2)),'fro');
history.stationarity = stationarity;
history.iterate_change = iterate_change;
history.orthogonality = orthogonality;
history.proximal_parameter = t;
history.stop_reason = stop_reason;
history.L_norm = L_norm;
fprintf('ManPG-Ada final: iter=%d flag=%d F=% .12e stat=%.3e orth=%.3e time=%.5f s\n',iter,succ_flag,F_val,stationarity,orthogonality,cpu_time);
end

function X = polar_retraction(Y)
[U,D,V] = svd(Y'*Y);
X = Y*(U*diag(1./sqrt(diag(D)))*V');
end

function [Z,inner_iterations,Lambda,residual,stop_flag] = ManPGSSN(X,t,B,threshold,tolerance,maxiter,Lambda,cg_floor,Dn,pDn)
[n,p] = size(X);

X_transpose = X';
X_Lambda = B+2*t*X*Lambda;
[Z,active,inactive] = soft_threshold(X_Lambda,threshold,p);
ZX = Z'*X;
R = ZX+ZX'-2*eye(p);
reduced_residual = pDn*R(:);
residual = norm(R,'fro');
regularization_factor = 0.2;
inner_iterations = 0;
stop_flag = 0;

while residual^2>tolerance
    if inner_iterations>=maxiter
        stop_flag = 1;
        break
    end

    regularization = regularization_factor*max(min(residual,0.1),1e-11);
    if p<15
        active_count = nnz(Z);
        if active_count>floor(p*(p+1)/2)
            G = zeros(p^2);
            for column_index = 1:p
                indices = (column_index-1)*p+1:column_index*p;
                G(indices,indices) = X'*(active(:,column_index).*X);
            end
            reduced_matrix = 4*t*(pDn*(G*Dn))+regularization*eye(p*(p+1)/2);
            direction = -Dn*(reduced_matrix\reduced_residual);
        else
            X_stack = zeros(active_count,p^2);
            first_row = 1;
            for column_index = 1:p
                rows = find(active(:,column_index));
                indices = (column_index-1)*p+1:column_index*p;
                last_row = first_row+numel(rows)-1;
                X_stack(first_row:last_row,indices) = X(rows,:);
                first_row = last_row+1;
            end
            V = X_stack*Dn;
            U = 4*t*(pDn*X_stack');
            direction = -Dn*(reduced_residual/regularization-U*((eye(active_count)+(V*U)/regularization)\(V*reduced_residual))/regularization^2);
        end
        direction = reshape(direction,p,p);
    else
        blocks = cell(p,1);
        for column_index = 1:p
            indices = active(:,column_index);
            if sum(indices)<n/2
                X_active = X(indices,:);
                blocks{column_index} = X_active'*X_active;
            else
                X_inactive = X_transpose(:,inactive(:,column_index));
                blocks{column_index} = eye(p)-X_inactive*X_inactive';
            end
        end
        operator = @(W) apply_operator(blocks,W,p,t,regularization);
        direction = conjugate_gradient(operator,-R,min(1e-4,1e-3*residual),cg_floor);
    end

    step = 1;
    X_direction = 2*t*X*direction;
    X_Lambda_trial = X_Lambda+step*X_direction;
    [Z,active,inactive] = soft_threshold(X_Lambda_trial,threshold,p);
    ZX = Z'*X;
    R_trial = ZX+ZX'-2*eye(p);
    residual_trial = norm(R_trial,'fro');

    while residual_trial^2>=residual^2*(1-0.001*step) && step>1e-3
        step = 0.5*step;
        X_Lambda_trial = X_Lambda+step*X_direction;
        [Z,active,inactive] = soft_threshold(X_Lambda_trial,threshold,p);
        ZX = Z'*X;
        R_trial = ZX+ZX'-2*eye(p);
        residual_trial = norm(R_trial,'fro');
    end

    Lambda = Lambda+step*direction;
    X_Lambda = X_Lambda_trial;
    R = R_trial;
    residual = residual_trial;
    reduced_residual = pDn*R(:);
    inner_iterations = inner_iterations+1;
end
end

function value = apply_operator(blocks,W,p,t,regularization)
value = zeros(size(W));
for column_index = 1:p
    value(:,column_index) = blocks{column_index}*W(:,column_index);
end
value = 2*t*(value+value')+regularization*W;
end

function x = conjugate_gradient(operator,b,tolerance,tolerance_floor)
tolerance = max(tolerance,tolerance_floor);
x = zeros(size(b));
r = b;
residual_squared = norm(r,'fro')^2;
if residual_squared<tolerance^2
    return
end
p = r;
for iteration = 1:numel(b)
    Ap = operator(p);
    alpha = residual_squared/sum(p.*Ap,'all');
    x = x+alpha*p;
    r = r-alpha*Ap;
    new_residual_squared = norm(r,'fro')^2;
    if new_residual_squared<tolerance^2
        return
    end
    p = r+(new_residual_squared/residual_squared)*p;
    residual_squared = new_residual_squared;
end
end

function [Z,active,inactive] = soft_threshold(B,threshold,p)
difference = abs(B)-threshold;
if p<15
    active = double(difference>0);
else
    active = difference>0;
end
Z = active.*sign(B).*difference;
inactive = difference<=0;
end
