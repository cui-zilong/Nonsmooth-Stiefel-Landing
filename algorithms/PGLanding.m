function [X_out,F_val,orth_err,sparsity,cpu_time,iter,succ_flag,history] = PGLanding(L,option)
%PGLANDING PGLanding method for sparse PCA.
% SPDX-License-Identifier: MIT
% The PGLanding algorithm is developed in the accompanying paper.
% Its semismooth Newton implementation is adapted from the MIT-licensed
% ManPG MATLAB package: https://github.com/chenshixiang/ManPG
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
zero_tol = option.zero_tol;
epsilon = option.epsilon;
lambda = option.lambda;
fixed_t_factor = option.fixed_t_factor;
verbose = option.verbose;
comparison_mode = option.comparison_mode;
if comparison_mode==1
    F_reference = option.reference_objective;
end

L_scale = abs(eigs(L,1,'largestabs'));
t = fixed_t_factor/L_scale;
[n,p] = size(X);
Ip = eye(p);
ssn.maxiter = option.ssn_maxiter;
ssn.tol = option.inner_tol_max;
ssn.armijo_c1 = 1e-3;
ssn.backtrack_tau = 0.5;
ssn.max_backtracks = 10;
ssn.min_step = 1e-3;
ssn.regularization_factor = 0.2;
ssn.matrix_free_threshold = 15;
inner_tol_min = option.inner_tol_min;
inner_tol_max = option.inner_tol_max;
inner_tol_forcing = option.inner_tol_forcing;
matrix_tol_factor = option.matrix_free_tolerance_factor;
matrix_tol_min = option.matrix_free_tolerance_min;
matrix_tol_max = option.matrix_free_tolerance_max;
inner_failure_tightening = option.inner_failure_tightening;

L_F = L_scale*sqrt(p*(1+epsilon))+norm(linear_term,'fro')+gamma*sqrt(n*p);
merit_sigma = 1.1*(lambda*L_F*sqrt(1+epsilon)+1e-3)/(2*lambda*(1-epsilon));
LX = L*X;
E = X'*X-Ip;
F_val = objective_value(X,LX,linear_term,gamma);
inner_tol = inner_tol_max;
solver_state = [];
iterate_change = inf;
orth_err = norm(E,'fro');
stat_residual = inf;
succ_flag = 0;
stop_reason = "maximum_subproblem_attempts";
outer_sweeps = 0;
accepted_updates = 0;
total_backtracks = 0;

for iter = 0:maxiter
    outer_sweeps = outer_sweeps+1;
    ssn.tol = inner_tol;
    ssn.matrix_free_tolerance = max(matrix_tol_min,min(matrix_tol_max,matrix_tol_factor*inner_tol));
    used_inner_tol = inner_tol;
    [V,solver_state] = PGLandingSSN(X,LX-linear_term,E+Ip,t,gamma,ssn,solver_state);
    stat_residual = norm(V,'fro')/t;
    orth_err = norm(E,'fro');
    combined_residual = max(stat_residual,orth_err);
    inner_tol = max(inner_tol_min,min(inner_tol_max,inner_tol_forcing*combined_residual));

    if verbose==1
        fprintf('[PGLanding %6d] F=% .12e stat=%.3e orth=%.3e step=%.3e inner=%.3e\n',iter,F_val,stat_residual,orth_err,iterate_change,used_inner_tol);
    end

    if comparison_mode==2
        converged = iterate_change<=tol && orth_err<=tol;
        current_reason = "step_and_feasibility_tol";
    elseif comparison_mode
        converged = F_val-F_reference<=tol && iterate_change<=tol && orth_err<=tol;
        current_reason = "objective_step_and_feasibility_tol";
    else
        converged = combined_residual<=tol;
        current_reason = "stationarity_and_feasibility_tol";
    end
    if converged
        succ_flag = 1;
        stop_reason = current_reason;
        break
    end
    if iter==maxiter
        break
    end

    LV = L*V;
    normal_correction = -lambda*X*E;
    direction = V+normal_correction;
    Ldirection = LV-lambda*LX*E;
    joint_measure = 1e-3*(t*stat_residual^2+orth_err);
    linear_change = sum(direction.*(linear_term-LX),'all');
    quadratic_change = -0.5*sum(direction.*Ldirection,'all');
    alpha = 1;
    backtracks = 0;

    while true
        X_trial = X+alpha*direction;
        LX_trial = LX+alpha*Ldirection;
        E_trial = X_trial'*X_trial-Ip;
        orth_trial = norm(E_trial,'fro');
        l1_change = gamma*sum(abs(X_trial)-abs(X),'all');
        objective_change = alpha*linear_change+alpha^2*quadratic_change+l1_change;
        accepted = orth_trial<=epsilon && objective_change+merit_sigma*(orth_trial-orth_err)<=-alpha*joint_measure;
        if accepted || alpha<=1e-4
            break
        end
        alpha = 0.5*alpha;
        backtracks = backtracks+1;
    end
    total_backtracks = total_backtracks+backtracks;

    if ~accepted
        inner_tol = max(inner_tol_min,inner_failure_tightening*used_inner_tol);
        continue
    end

    X_old = X;
    X = X_trial;
    LX = LX_trial;
    E = E_trial;
    F_val = F_val+objective_change;
    iterate_change = norm(X-X_old,'fro');
    accepted_updates = accepted_updates+1;
end

cpu_time = toc(start_clock);
X_out = X;
iter = accepted_updates;
sparsity = nnz(abs(X)<=zero_tol)/numel(X);
history.stationarity = stat_residual;
history.iterate_change = iterate_change;
history.outer_sweeps = outer_sweeps;
history.accepted_outer_updates = accepted_updates;
history.outer_backtracks = total_backtracks;
history.stop_reason = stop_reason;
history.L_norm = L_scale;
history.t = t;
fprintf('PGLanding final: iter=%d flag=%d F=% .12e stat=%.3e orth=%.3e time=%.5f s\n',iter,succ_flag,F_val,stat_residual,orth_err,cpu_time);
end

function value = objective_value(X,LX,linear_term,gamma)
value = -0.5*sum(X(:).*LX(:))+sum(linear_term(:).*X(:))+gamma*sum(abs(X(:)));
end

function [V,state] = PGLandingSSN(X,LX,XTX,t,gamma,settings,state)
[n,p] = size(X);

if isempty(state)
    state.Lam = zeros(p);
    state.Dn = duplication_matrix(p);
    state.metric_diagonal = full(sum(state.Dn.^2,1))';
    state.metric_square_root = sqrt(state.metric_diagonal);
    if p<settings.matrix_free_threshold
        q = size(state.Dn,2);
        state.pDn = spdiags(1./state.metric_diagonal,0,q,q)*state.Dn';
        state.Dn_blocks = cell(p,1);
        for column = 1:p
            rows = (column-1)*p+(1:p);
            state.Dn_blocks{column} = state.Dn(rows,:);
        end
    end
end

Lam = state.Lam;
Dn = state.Dn;
metric_diagonal = state.metric_diagonal;
metric_square_root = state.metric_square_root;
q = size(Dn,2);
base_center = X+t*LX;
threshold = t*gamma;
center = base_center+2*t*X*Lam;
[Y,R,active] = proximal_state(X,XTX,center,threshold);
residual = norm(R,'fro');
previous_active = [];
active_gram_blocks = [];
gram_diagonal = [];
if p<settings.matrix_free_threshold
    block_times_duplication = zeros(p*p,q);
end
iterations_done = 0;

for newton_iteration = 1:settings.maxiter
    if residual<=settings.tol
        break
    end
    iterations_done = newton_iteration;

    if isempty(previous_active) || ~isequal(active,previous_active)
        active_gram_blocks = cell(p,1);
        gram_diagonal = zeros(p);
        for column = 1:p
            active_rows = active(:,column);
            if nnz(active_rows)<=n/2
                X_selected = X(active_rows,:);
                block_gram = X_selected'*X_selected;
            else
                X_selected = X(~active_rows,:);
                block_gram = XTX-X_selected'*X_selected;
            end
            active_gram_blocks{column} = block_gram;
            gram_diagonal(:,column) = diag(block_gram);
            if p<settings.matrix_free_threshold
                rows = (column-1)*p+(1:p);
                block_times_duplication(rows,:) = block_gram*state.Dn_blocks{column};
            end
        end
        previous_active = active;
    end

    regularization = settings.regularization_factor*max(min(residual,0.1),1e-11);
    if p<settings.matrix_free_threshold
        reduced_residual = state.pDn*R(:);
        system_matrix = 4*t*(state.pDn*block_times_duplication)+regularization*eye(q);
        [lower_factor,upper_factor] = lu(system_matrix);
        delta_coordinates = -(upper_factor\(lower_factor\reduced_residual));
    else
        right_hand_side = -(Dn'*R(:))./metric_square_root;
        operator = @(z) jacobian_action(z,active_gram_blocks,t,regularization,Dn,metric_square_root,p);
        if norm(right_hand_side)<=settings.matrix_free_tolerance
            scaled_delta = zeros(q,1);
        else
            block_diagonal = gram_diagonal(:);
            preconditioner = 4*t*((Dn'*block_diagonal)./metric_diagonal)+regularization;
            preconditioner = max(preconditioner,max(regularization,eps));
            scaled_delta = PGLandingMINRES(operator,right_hand_side,settings.matrix_free_tolerance,q,preconditioner);
        end
        delta_coordinates = scaled_delta./metric_square_root;
    end

    delta_Lam = reshape(Dn*delta_coordinates,p,p);
    center_direction = 2*t*X*delta_Lam;
    residual_square = residual^2;
    alpha = 1;
    best_residual_square = inf;

    for backtrack = 0:settings.max_backtracks
        Lam_trial = Lam+alpha*delta_Lam;
        center_trial = center+alpha*center_direction;
        [Y_trial,R_trial,active_trial] = proximal_state(X,XTX,center_trial,threshold);
        residual_trial = norm(R_trial,'fro');
        residual_square_trial = residual_trial^2;
        if residual_square_trial<best_residual_square
            best_residual_square = residual_square_trial;
            best_Y = Y_trial;
            best_R = R_trial;
            best_active = active_trial;
            best_Lam = Lam_trial;
            best_center = center_trial;
        end
        accepted = residual_square_trial<residual_square*(1-settings.armijo_c1*alpha) || residual_trial<=settings.tol || alpha<=settings.min_step;
        if accepted
            break
        end
        if backtrack<settings.max_backtracks
            alpha = settings.backtrack_tau*alpha;
        end
    end

    if accepted
        Lam = Lam_trial;
        center = center_trial;
        Y = Y_trial;
        R = R_trial;
        active = active_trial;
        residual = residual_trial;
    elseif best_residual_square<residual_square
        Lam = best_Lam;
        center = best_center;
        Y = best_Y;
        R = best_R;
        active = best_active;
        residual = sqrt(best_residual_square);
    else
        break
    end
end

V = Y-X;
state.Lam = Lam;
state.newton_iterations = iterations_done;
state.residual = norm(R,'fro');
end

function output = jacobian_action(scaled_coordinates,active_gram_blocks,t,regularization,Dn,metric_square_root,p)
coordinates = scaled_coordinates./metric_square_root;
delta_Lam = reshape(Dn*coordinates,p,p);
matrix_product = zeros(p);
for column = 1:p
    matrix_product(:,column) = active_gram_blocks{column}*delta_Lam(:,column);
end
output = 4*t*((Dn'*matrix_product(:))./metric_square_root)+regularization*scaled_coordinates;
end

function [Y,R,active] = proximal_state(X,XTX,center,threshold)
Y = sign(center).*max(abs(center)-threshold,0);
YtX = Y'*X;
R = YtX+YtX'-2*XTX;
active = Y~=0;
end

function D = duplication_matrix(p)
q = p*(p+1)/2;
row_indices = zeros(p*p,1);
column_indices = zeros(p*p,1);
coordinate = 0;
nonzero_index = 0;
for column = 1:p
    for row = column:p
        coordinate = coordinate+1;
        nonzero_index = nonzero_index+1;
        row_indices(nonzero_index) = sub2ind([p,p],row,column);
        column_indices(nonzero_index) = coordinate;
        if row~=column
            nonzero_index = nonzero_index+1;
            row_indices(nonzero_index) = sub2ind([p,p],column,row);
            column_indices(nonzero_index) = coordinate;
        end
    end
end
D = sparse(row_indices(1:nonzero_index),column_indices(1:nonzero_index),ones(nonzero_index,1),p*p,q);
end


function x = PGLandingMINRES(A,b,tol,maxiter,preconditioner)
b = b(:);
inverse_sqrt_diagonal = 1./sqrt(preconditioner(:));
apply_A = @(z) inverse_sqrt_diagonal.*A(inverse_sqrt_diagonal.*z);
y = zeros(size(b));
residual = inverse_sqrt_diagonal.*b;
beta1 = norm(residual);
if beta1==0
    x = zeros(size(b));
    return
end

residual_old = zeros(size(b));
beta_old = 0;
beta = beta1;
dbar = 0;
epsilon_minres = 0;
phi_bar = beta1;
cosine = -1;
sine = 0;
direction = zeros(size(b));
direction_old = zeros(size(b));
residual_direction = residual/beta1;

for k = 1:maxiter
    lanczos_vector = residual/beta;
    next_residual = apply_A(lanczos_vector);
    if k>1
        next_residual = next_residual-(beta/beta_old)*residual_old;
    end
    alpha = real(lanczos_vector'*next_residual);
    next_residual = next_residual-(alpha/beta)*residual;
    residual_old = residual;
    residual = next_residual;
    beta_old = beta;
    beta = norm(residual);

    old_epsilon = epsilon_minres;
    delta = cosine*dbar+sine*alpha;
    gbar = sine*dbar-cosine*alpha;
    epsilon_minres = sine*beta;
    dbar = -cosine*beta;
    gamma = hypot(gbar,beta);
    if gamma==0
        break
    end

    cosine = gbar/gamma;
    sine = beta/gamma;
    phi = cosine*phi_bar;
    phi_bar = sine*phi_bar;
    direction_older = direction_old;
    direction_old = direction;
    direction = (lanczos_vector-old_epsilon*direction_older-delta*direction_old)/gamma;
    y = y+phi*direction;

    if beta>0
        next_lanczos_vector = residual/beta;
        residual_direction = sine*residual_direction-cosine*next_lanczos_vector;
        estimated_residual = abs(phi_bar)*norm(sqrt(preconditioner(:)).*residual_direction);
    else
        estimated_residual = 0;
    end

    if estimated_residual<=tol || beta==0
        candidate = inverse_sqrt_diagonal.*y;
        if norm(b-A(candidate))<=tol || beta==0
            break
        end
    end
end

x = inverse_sqrt_diagonal.*y;
end
