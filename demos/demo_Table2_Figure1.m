clear;
clc;
close all;

%% Section 6.2: Table 2 and Figure 1
% Part I: normal-correction parameters. Part II: convergence histories.
% The local RieSub loop follows the ManPG-derived RieSub.m (MIT);
% see THIRD_PARTY_NOTICE.md.

demo_directory = fileparts(mfilename('fullpath'));
repository_root = fileparts(demo_directory);
addpath(repository_root,'-begin');
setup_paths;

%% Common sparse-PCA instance
n = 500;
m = 500;
p = 20;
gamma = 0.5;
data_seed = 0;
[L,X_random] = Gen_SPCA_Random(n,p,m,data_seed);
Ip = eye(p);

%% Common settings
run_only_part_two = false; % Set true to generate Figure 1 only.
initialization_iterations = 250;
maxiter = 50000;
eta0 = 1e-3;
geometric_beta = 0.999;
zero_tol = 1e-5;
disabled_tolerance = -1;
number_of_repetitions = 10; % Timed runs per Table 2 configuration.
history_stride = 100;

fprintf('============================================================\n');
fprintf('RieSub--SubLanding sparse-PCA comparison\n');
fprintf('n=%d, m=%d, p=%d, gamma=%.2f, seed=%d\n',n,m,p,gamma,data_seed);
fprintf('eta0=%.1e, maxiter=%d, repetitions=%d\n',eta0,maxiter,number_of_repetitions);
fprintf('============================================================\n');

%% Common initial point
initialization_option = struct();
initialization_option.phi_init = X_random;
initialization_option.gamma = gamma;
initialization_option.maxiter = initialization_iterations;
initialization_option.tol = disabled_tolerance;
initialization_option.eta0 = 1;
initialization_option.zero_tol = zero_tol;
initialization_option.stepsize_rule = 'power_three_quarters';
initialization_option.stepsize_beta = geometric_beta;
initialization_option.verbose = 0;
initialization_option.comparison_mode = false;
[X_init,F_init,sparsity_init,initialization_time,initialization_iter,initialization_flag] = RieSub(L,initialization_option);
orthogonality_init = norm(X_init'*X_init-Ip,'fro');

fprintf('\nCommon initial point\n');
fprintf('Iterations    : %d\n',initialization_iter);
fprintf('Objective     : %.12e\n',F_init);
fprintf('Orthogonality : %.3e\n',orthogonality_init);
fprintf('Sparsity      : %.4f\n',sparsity_init);
fprintf('Time          : %.5f s\n',initialization_time);
fprintf('Flag          : %d\n',initialization_flag);

%% High-accuracy ManPG-Ada reference
reference_option = struct();
reference_option.phi_init = X_init;
reference_option.gamma = gamma;
reference_option.maxiter = 100000;
reference_option.tol = 1e-10;
reference_option.inner_iter = 100;
reference_option.inner_residual_tolerance = 1e-12;
reference_option.zero_tol = zero_tol;
reference_option.verbose = 0;
reference_option.comparison_mode = false;
[X_ref,F_ref,sparsity_ref,reference_time,reference_iter,reference_flag,reference_linesearch,reference_mean_ssn,reference_history] = ManPGAda(L,reference_option);
reference_orthogonality = norm(X_ref'*X_ref-Ip,'fro');

if reference_flag~=1 || ~isfinite(reference_history.stationarity) || reference_history.stationarity>reference_option.tol || reference_orthogonality>1e-9
    error('ManPG-Ada did not meet the reference tolerances.');
end

fprintf('\nManPG-Ada reference\n');
fprintf('Objective     : %.12e\n',F_ref);
fprintf('Stationarity  : %.3e\n',reference_history.stationarity);
fprintf('Orthogonality : %.3e\n',reference_orthogonality);
fprintf('Sparsity      : %.4f\n',sparsity_ref);
fprintf('Iterations    : %d\n',reference_iter);
fprintf('Time          : %.5f s\n',reference_time);
fprintf('Line searches : %d\n',reference_linesearch);
fprintf('Mean SSN      : %.3f\n',reference_mean_ssn);

%% Part I: Table 2
if ~run_only_part_two
polynomial_rules = {'power_half'};
polynomial_exponents = 0.5;
sublanding_alphas = [1,1,1,0,0,0];
sublanding_lambdas = [5,50,500,0.25,0.5,0.75];
configurations_per_rule = 1+numel(sublanding_alphas);
number_of_configurations = numel(polynomial_rules)*configurations_per_rule;

configuration_method = strings(number_of_configurations,1);
configuration_rule = strings(number_of_configurations,1);
configuration_exponent = nan(number_of_configurations,1);
configuration_alpha = nan(number_of_configurations,1);
configuration_lambda = nan(number_of_configurations,1);
configuration_index = 0;

for rule_index = 1:numel(polynomial_rules)
    configuration_index = configuration_index+1;
    configuration_method(configuration_index) = "RieSub";
    configuration_rule(configuration_index) = string(polynomial_rules{rule_index});
    configuration_exponent(configuration_index) = polynomial_exponents(rule_index);
    for parameter_index = 1:numel(sublanding_alphas)
        configuration_index = configuration_index+1;
        configuration_method(configuration_index) = "SubLanding";
        configuration_rule(configuration_index) = string(polynomial_rules{rule_index});
        configuration_exponent(configuration_index) = polynomial_exponents(rule_index);
        configuration_alpha(configuration_index) = sublanding_alphas(parameter_index);
        configuration_lambda(configuration_index) = sublanding_lambdas(parameter_index);
    end
end

benchmark_time = nan(number_of_configurations,number_of_repetitions);
benchmark_iter = nan(number_of_configurations,1);
benchmark_flag = nan(number_of_configurations,1);
benchmark_objective = nan(number_of_configurations,1);
benchmark_orthogonality = nan(number_of_configurations,1);
benchmark_sparsity = nan(number_of_configurations,1);
benchmark_backtracks = nan(number_of_configurations,1);

fprintf('\nShort untimed code-path warm-up\n');
timing_warmup_iterations = min(200,maxiter);
for current_configuration = 1:number_of_configurations
    run_fixed_configuration(L,X_init,gamma,timing_warmup_iterations,eta0,zero_tol,disabled_tolerance,char(configuration_method(current_configuration)),char(configuration_rule(current_configuration)),configuration_alpha(current_configuration),configuration_lambda(current_configuration));
end

fprintf('\nPart I: repeated timing for Table 2\n');
for repetition = 1:number_of_repetitions
    if mod(repetition,2)==1
        execution_order = 1:number_of_configurations;
        order_name = 'forward';
    else
        execution_order = number_of_configurations:-1:1;
        order_name = 'reverse';
    end
    fprintf('Repetition %d/%d (%s order)\n',repetition,number_of_repetitions,order_name);
    for execution_position = 1:number_of_configurations
        current_configuration = execution_order(execution_position);
        [X_current,F_current,orth_current,sparsity_current,time_current,iter_current,flag_current,backtracks_current] = run_fixed_configuration(L,X_init,gamma,maxiter,eta0,zero_tol,disabled_tolerance,char(configuration_method(current_configuration)),char(configuration_rule(current_configuration)),configuration_alpha(current_configuration),configuration_lambda(current_configuration));
        benchmark_time(current_configuration,repetition) = time_current;
        if repetition==1
            LX_current = L*X_current;
            benchmark_iter(current_configuration) = iter_current;
            benchmark_flag(current_configuration) = flag_current;
            benchmark_objective(current_configuration) = -0.5*sum(X_current.*LX_current,'all')+gamma*sum(abs(X_current),'all');
            benchmark_orthogonality(current_configuration) = norm(X_current'*X_current-Ip,'fro');
            benchmark_sparsity(current_configuration) = nnz(abs(X_current)<=zero_tol)/numel(X_current);
            benchmark_backtracks(current_configuration) = backtracks_current;
            if abs(F_current-benchmark_objective(current_configuration))>1e-10*max(1,abs(benchmark_objective(current_configuration)))
                warning('The reported and recomputed objectives differ for configuration %d.',current_configuration);
            end
            if abs(orth_current-benchmark_orthogonality(current_configuration))>1e-10
                warning('The reported and recomputed orthogonality values differ for configuration %d.',current_configuration);
            end
            if abs(sparsity_current-benchmark_sparsity(current_configuration))>eps
                warning('The reported and recomputed sparsity values differ for configuration %d.',current_configuration);
            end
        end
    end
end

Method = configuration_method;
StepsizeRule = configuration_rule;
Exponent = configuration_exponent;
Alpha = configuration_alpha;
Lambda = configuration_lambda;
Repetitions = repmat(number_of_repetitions,number_of_configurations,1);
Iter = benchmark_iter;
Flag = benchmark_flag;
Objective = benchmark_objective;
ObjectiveGap = benchmark_objective-F_ref;
OrthErr = benchmark_orthogonality;
Sparsity = benchmark_sparsity;
MeanTimeSeconds = mean(benchmark_time,2);
MeanTimePerIterMs = 1000*MeanTimeSeconds./max(Iter,1);
TotalBacktracks = benchmark_backtracks;
parameter_results = table(Method,StepsizeRule,Exponent,Alpha,Lambda,Repetitions,Iter,Flag,Objective,ObjectiveGap,OrthErr,Sparsity,MeanTimeSeconds,MeanTimePerIterMs,TotalBacktracks);

fprintf('\n================ PARAMETER RESULTS ================\n');
disp(parameter_results);
writetable(parameter_results,fullfile(demo_directory,'SPCA_RieSub_SubLanding_parameter_results.csv'));
end

%% Part II: selected-parameter convergence histories
selected_alpha = 0;
selected_lambda = 0.5;
history_rules = {'power_half','geometric'};
% MATLAB update k=1 corresponds to the paper's k=0.
history_eta0 = [eta0,eta0];
history_stepsize_labels = ["eta_k=1e-3*(k+1)^(-1/2)","eta_k=1e-3*(0.999)^k"];
number_of_curves = 4;
curve_method = ["RieSub";"SubLanding";"RieSub";"SubLanding"];
curve_rule_index = [1;1;2;2];
curve_X = cell(number_of_curves,1);
curve_F = nan(number_of_curves,1);
curve_orthogonality = nan(number_of_curves,1);
curve_sparsity = nan(number_of_curves,1);
curve_time = nan(number_of_curves,1);
curve_iter = nan(number_of_curves,1);
curve_flag = nan(number_of_curves,1);
curve_backtracks = zeros(number_of_curves,1);
curve_history = cell(number_of_curves,1);

fprintf('\nPart II: instrumented runs for convergence figures\n');
for curve_index = 1:number_of_curves
    rule_index = curve_rule_index(curve_index);
    history_option = struct();
    history_option.phi_init = X_init;
    history_option.gamma = gamma;
    history_option.maxiter = maxiter;
    history_option.eta0 = history_eta0(rule_index);
    history_option.stepsize_rule = history_rules{rule_index};
    history_option.stepsize_beta = geometric_beta;
    history_option.history_stride = history_stride;
    history_option.reference_X = X_ref;
    history_option.zero_tol = zero_tol;
    if curve_method(curve_index)=="RieSub"
        [curve_X{curve_index},curve_F(curve_index),curve_orthogonality(curve_index),curve_sparsity(curve_index),curve_time(curve_index),curve_iter(curve_index),curve_flag(curve_index),curve_history{curve_index}] = run_riesub_history(L,history_option);
    else
        history_option.alpha = selected_alpha;
        history_option.lambda = selected_lambda;
        history_option.epsilon = 0.5;
        history_option.safe_backtrack_tau = 0.5;
        history_option.safe_max_backtracks = 60;
        history_option.eta_min = 1e-8;
        [curve_X{curve_index},curve_F(curve_index),curve_orthogonality(curve_index),curve_sparsity(curve_index),curve_time(curve_index),curve_iter(curve_index),curve_flag(curve_index),curve_backtracks(curve_index),curve_history{curve_index}] = run_sublanding_history(L,history_option);
    end
end

relative_objective_denominator = 1+abs(F_ref);
iteration_history = cell(number_of_curves,1);
time_history = cell(number_of_curves,1);
distance_history = cell(number_of_curves,1);
orthogonality_history = cell(number_of_curves,1);

for curve_index = 1:number_of_curves
    iteration_history{curve_index} = curve_history{curve_index}.iteration;
    time_history{curve_index} = curve_history{curve_index}.time;
    distance_history{curve_index} = curve_history{curve_index}.distance;
    orthogonality_history{curve_index} = curve_history{curve_index}.feasibility;
end

distance_floor = 1e-10;
orthogonality_floor = 1e-15;
orthogonality_ceiling = 1e-3;
right_padding_factor = 2.0;
distance_iteration_limit = find_display_limit(iteration_history,distance_history,distance_floor,right_padding_factor);
distance_time_limit = find_display_limit(time_history,distance_history,distance_floor,right_padding_factor);
orthogonality_iteration_limit = find_display_limit(iteration_history,orthogonality_history,orthogonality_floor,right_padding_factor);
orthogonality_time_limit = find_display_limit(time_history,orthogonality_history,orthogonality_floor,right_padding_factor);

figure_directory = fullfile(demo_directory,'figures_SPCA_RieSub_SubLanding');
if ~isfolder(figure_directory)
    mkdir(figure_directory);
end

riesub_color = [0.80,0.18,0.14];
sublanding_color = [0.00,0.36,0.70];
curve_colors = [riesub_color;sublanding_color;riesub_color;sublanding_color];
curve_line_styles = {'-','-','--','--'};
curve_markers = {'^','s','^','s'};
curve_line_widths = [2.0,1.8,2.0,1.8];
distance_y_label = '$\|X_k-X_{\rm ref}\|_{\mathrm{F}}$';
orthogonality_y_label = '$\|X_k^\top X_k-I_p\|_{\mathrm{F}}$';
figure_handle = make_combined_history_figure(iteration_history,time_history,distance_history,orthogonality_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,distance_y_label,orthogonality_y_label,distance_floor,orthogonality_floor,orthogonality_ceiling,distance_iteration_limit,orthogonality_iteration_limit,distance_time_limit,orthogonality_time_limit);
export_combined_figure(figure_handle,fullfile(figure_directory,'RieSub_SubLanding_convergence.png'),fullfile(figure_directory,'RieSub_SubLanding_convergence.pdf'));

Method = ["ManPG-Ada reference";curve_method];
Stepsize = ["-";reshape(history_stepsize_labels(curve_rule_index),[],1)];
Alpha = [NaN;NaN;selected_alpha;NaN;selected_alpha];
Lambda = [NaN;NaN;selected_lambda;NaN;selected_lambda];
Iter = [reference_iter;curve_iter];
Flag = [reference_flag;curve_flag];
Objective = [F_ref;curve_F];
RelativeObjectiveError = [0;abs(curve_F-F_ref)/relative_objective_denominator];
OrthErr = [reference_orthogonality;curve_orthogonality];
Sparsity = [sparsity_ref;curve_sparsity];
TimeSeconds = [reference_time;curve_time];
TimePerIterMs = 1000*TimeSeconds./max(Iter,1);
TotalBacktracks = [NaN;curve_backtracks];
convergence_results = table(Method,Stepsize,Alpha,Lambda,Iter,Flag,Objective,RelativeObjectiveError,OrthErr,Sparsity,TimeSeconds,TimePerIterMs,TotalBacktracks);

fprintf('\n================ CONVERGENCE RESULTS ================\n');
disp(convergence_results);
writetable(convergence_results,fullfile(demo_directory,'SPCA_RieSub_SubLanding_convergence_results.csv'));
if ~run_only_part_two
    fprintf('\nSaved parameter table: SPCA_RieSub_SubLanding_parameter_results.csv\n');
end
fprintf('Saved convergence table: SPCA_RieSub_SubLanding_convergence_results.csv\n');
fprintf('Saved the combined four-panel figure in: %s\n',figure_directory);


function [X,F,orthogonality,sparsity,cpu_time,iter,flag,backtracks] = run_fixed_configuration(L,X_init,gamma,maxiter,eta0,zero_tol,tol,method,stepsize_rule,alpha,lambda)
option = struct();
option.phi_init = X_init;
option.gamma = gamma;
option.maxiter = maxiter;
option.tol = tol;
option.eta0 = eta0;
option.zero_tol = zero_tol;
option.stepsize_rule = stepsize_rule;
option.stepsize_beta = 0.999;
option.verbose = 0;
option.comparison_mode = false;
if strcmp(method,'RieSub')
    [X,F,sparsity,cpu_time,iter,flag] = RieSub(L,option);
    orthogonality = norm(X'*X-eye(size(X,2)),'fro');
    backtracks = 0;
else
    option.alpha = alpha;
    option.lambda = lambda;
    option.epsilon = 0.5;
    option.safe_backtrack_tau = 0.5;
    option.safe_max_backtracks = 60;
    [X,F,orthogonality,sparsity,cpu_time,iter,flag,history] = SubLanding(L,option);
    backtracks = history.total_backtracks;
end
end

function [X,F,orthogonality,sparsity,cpu_time,iter,flag,history] = run_riesub_history(L,option)
start_clock = tic;
X = option.phi_init;
gamma = option.gamma;
maxiter = option.maxiter;
eta0 = option.eta0;
stepsize_rule = option.stepsize_rule;
stepsize_beta = option.stepsize_beta;
history_stride = option.history_stride;
Ip = eye(size(X,2));
LX = L*X;
maximum_records = ceil(maxiter/history_stride)+2;
history.iteration = nan(maximum_records,1);
history.objective = nan(maximum_records,1);
history.feasibility = nan(maximum_records,1);
history.distance = nan(maximum_records,1);
history.time = nan(maximum_records,1);
record_index = 1;
history.iteration(record_index) = 0;
history.objective(record_index) = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
history.feasibility(record_index) = norm(X'*X-Ip,'fro');
history.distance(record_index) = norm(X-option.reference_X,'fro');
history.time(record_index) = 0;
flag = 0;
iter = 0;
for k = 1:maxiter
    negative_subgradient = LX-gamma*sign(X);
    XG = X'*negative_subgradient;
    direction = negative_subgradient-0.5*X*(XG+XG');
    eta = history_stepsize(eta0,stepsize_rule,stepsize_beta,k);
    X_trial = X+eta*direction;
    Gram = X_trial'*X_trial;
    Gram = 0.5*(Gram+Gram');
    [U,D,V] = svd(Gram);
    singular_values = max(diag(D),eps(class(X)));
    X = X_trial*(U*diag(1./sqrt(singular_values))*V');
    LX = L*X;
    iter = k;
    if mod(k,history_stride)==0 || k==maxiter
        record_index = record_index+1;
        history.iteration(record_index) = k;
        history.objective(record_index) = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
        history.feasibility(record_index) = norm(X'*X-Ip,'fro');
        history.distance(record_index) = norm(X-option.reference_X,'fro');
        history.time(record_index) = toc(start_clock);
    end
end
cpu_time = toc(start_clock);
F = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
orthogonality = norm(X'*X-Ip,'fro');
sparsity = nnz(abs(X)<=option.zero_tol)/numel(X);
history.iteration = history.iteration(1:record_index);
history.objective = history.objective(1:record_index);
history.feasibility = history.feasibility(1:record_index);
history.distance = history.distance(1:record_index);
history.time = history.time(1:record_index);
end

function [X,F,orthogonality,sparsity,cpu_time,iter,flag,total_backtracks,history] = run_sublanding_history(L,option)
start_clock = tic;
X = option.phi_init;
gamma = option.gamma;
maxiter = option.maxiter;
eta0 = option.eta0;
alpha = option.alpha;
lambda = option.lambda;
stepsize_rule = option.stepsize_rule;
stepsize_beta = option.stepsize_beta;
history_stride = option.history_stride;
Ip = eye(size(X,2));
LX = L*X;
XtX = X'*X;
E = XtX-Ip;
subgradient = -LX+gamma*sign(X);
AtX = subgradient'*X;
tangent_gradient = subgradient*XtX-X*AtX;
maximum_records = ceil(maxiter/history_stride)+2;
history.iteration = nan(maximum_records,1);
history.objective = nan(maximum_records,1);
history.feasibility = nan(maximum_records,1);
history.distance = nan(maximum_records,1);
history.time = nan(maximum_records,1);
record_index = 1;
history.iteration(record_index) = 0;
history.objective(record_index) = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
history.feasibility(record_index) = norm(E,'fro');
history.distance(record_index) = norm(X-option.reference_X,'fro');
history.time(record_index) = 0;
flag = 0;
iter = 0;
total_backtracks = 0;
for k = 1:maxiter
    eta = history_stepsize(eta0,stepsize_rule,stepsize_beta,k);
    normal_base = X*E;
    accepted = false;
    backtracks = 0;
    while backtracks<=option.safe_max_backtracks
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
        trial_orthogonality = norm(trial_E,'fro');
        if trial_orthogonality<=option.epsilon
            accepted = true;
            break
        end
        eta = option.safe_backtrack_tau*eta;
        backtracks = backtracks+1;
        if eta<option.eta_min
            break
        end
    end
    total_backtracks = total_backtracks+backtracks;
    if ~accepted
        flag = 2;
        break
    end
    X = X_trial;
    LX = L*X;
    XtX = trial_XtX;
    E = trial_E;
    subgradient = -LX+gamma*sign(X);
    AtX = subgradient'*X;
    tangent_gradient = subgradient*XtX-X*AtX;
    iter = k;
    if mod(k,history_stride)==0 || k==maxiter
        record_index = record_index+1;
        history.iteration(record_index) = k;
        history.objective(record_index) = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
        history.feasibility(record_index) = norm(E,'fro');
        history.distance(record_index) = norm(X-option.reference_X,'fro');
        history.time(record_index) = toc(start_clock);
    end
end
if history.iteration(record_index)~=iter
    record_index = record_index+1;
    history.iteration(record_index) = iter;
    history.objective(record_index) = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
    history.feasibility(record_index) = norm(E,'fro');
    history.distance(record_index) = norm(X-option.reference_X,'fro');
    history.time(record_index) = toc(start_clock);
end
cpu_time = toc(start_clock);
F = -0.5*sum(X.*LX,'all')+gamma*sum(abs(X),'all');
orthogonality = norm(E,'fro');
sparsity = nnz(abs(X)<=option.zero_tol)/numel(X);
history.iteration = history.iteration(1:record_index);
history.objective = history.objective(1:record_index);
history.feasibility = history.feasibility(1:record_index);
history.distance = history.distance(1:record_index);
history.time = history.time(1:record_index);
end

function eta = history_stepsize(eta0,stepsize_rule,stepsize_beta,k)
if strcmp(stepsize_rule,'power_half')
    eta = eta0/k^0.5;
elseif strcmp(stepsize_rule,'power_three_quarters')
    eta = eta0/k^0.75;
else
    eta = eta0*stepsize_beta^(k-1);
end
end

function x_limit = find_display_limit(x_history,y_history,y_floor,padding_factor)
full_limit = 0;
stable_entry = nan(numel(x_history),1);
stable_count = 0;
for curve_index = 1:numel(x_history)
    current_x = x_history{curve_index}(:);
    current_y = y_history{curve_index}(:);
    valid = isfinite(current_x) & isfinite(current_y) & current_x>=0 & current_y>=0;
    current_x = current_x(valid);
    current_y = current_y(valid);
    if isempty(current_x)
        continue
    end
    full_limit = max(full_limit,max(current_x));
    last_above = find(current_y>y_floor,1,'last');
    if isempty(last_above)
        first_below = 1;
    else
        first_below = last_above+1;
    end
    if first_below<=numel(current_y)-2 && all(current_y(first_below:end)<=y_floor)
        stable_count = stable_count+1;
        stable_entry(stable_count) = current_x(first_below);
    end
end
stable_entry = stable_entry(1:stable_count);
if isempty(stable_entry)
    x_limit = full_limit;
else
    x_limit = min(padding_factor*min(stable_entry),full_limit);
end
if ~isfinite(x_limit) || x_limit<=0
    x_limit = 1;
end
end

function figure_handle = make_combined_history_figure(iteration_history,time_history,distance_history,orthogonality_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,distance_y_label,orthogonality_y_label,distance_floor,orthogonality_floor,orthogonality_ceiling,distance_iteration_limit,orthogonality_iteration_limit,distance_time_limit,orthogonality_time_limit)
figure_handle = figure('Visible','off','Color','w','Units','inches','Position',[1,1,13.2,9.8]);
layout_handle = tiledlayout(figure_handle,2,2,'TileSpacing','compact','Padding','compact');
distance_iteration_axes = nexttile(layout_handle,1);
legend_handles = plot_history_axes(distance_iteration_axes,iteration_history,distance_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,'Iteration',distance_y_label,distance_floor,NaN,distance_iteration_limit,false);
orthogonality_iteration_axes = nexttile(layout_handle,2);
plot_history_axes(orthogonality_iteration_axes,iteration_history,orthogonality_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,'Iteration',orthogonality_y_label,orthogonality_floor,orthogonality_ceiling,orthogonality_iteration_limit,true);
distance_time_axes = nexttile(layout_handle,3);
plot_history_axes(distance_time_axes,time_history,distance_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,'CPU time (s)',distance_y_label,distance_floor,NaN,distance_time_limit,false);
orthogonality_time_axes = nexttile(layout_handle,4);
plot_history_axes(orthogonality_time_axes,time_history,orthogonality_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,'CPU time (s)',orthogonality_y_label,orthogonality_floor,orthogonality_ceiling,orthogonality_time_limit,true);
legend_labels = {'$\mathrm{RieSub},\quad \eta_k=10^{-3}(k+1)^{-1/2}\qquad$','$\mathrm{SubLanding},\quad \eta_k=10^{-3}(k+1)^{-1/2}\quad$','$\mathrm{RieSub},\quad \eta_k=10^{-3}(0.999)^k\qquad$','$\mathrm{SubLanding},\quad \eta_k=10^{-3}(0.999)^k\quad$'};
legend_handle = legend(distance_iteration_axes,legend_handles,legend_labels,'Interpreter','latex','Orientation','horizontal','NumColumns',2,'Box','on','FontSize',20);
legend_handle.Color = 'white';
legend_handle.TextColor = 'black';
legend_handle.EdgeColor = 'black';
legend_handle.ItemTokenSize = [44,22];
legend_handle.Layout.Tile = 'north';
end

function line_handles = plot_history_axes(axes_handle,x_history,y_history,curve_colors,curve_line_styles,curve_markers,curve_line_widths,x_label,y_label,y_lower,y_upper,x_upper,mark_initial_point)
hold(axes_handle,'on');
box(axes_handle,'on');
grid(axes_handle,'on');
set(axes_handle,'Color','white','XColor','black','YColor','black','XScale','linear','YScale','log','FontSize',20,'TickLabelInterpreter','latex','LineWidth',1.2,'GridColor',[0.15,0.15,0.15],'GridAlpha',0.15,'XMinorGrid','off','YMinorGrid','off');
line_handles = gobjects(numel(x_history),1);
for curve_index = 1:numel(x_history)
    current_x = x_history{curve_index};
    current_y = y_history{curve_index};
    visible = find(isfinite(current_x) & isfinite(current_y) & current_x<=x_upper & current_y>=y_lower);
    marker_count = min(8,numel(visible));
    marker_indices = [];
    if marker_count>0
        marker_positions = unique(round(linspace(2,numel(visible),marker_count)));
        marker_positions = marker_positions(marker_positions>=1 & marker_positions<=numel(visible));
        marker_indices = visible(marker_positions);
    end
    line_handles(curve_index) = plot(axes_handle,current_x,current_y,'Color',curve_colors(curve_index,:),'LineStyle',curve_line_styles{curve_index},'LineWidth',curve_line_widths(curve_index),'Marker',curve_markers{curve_index},'MarkerIndices',marker_indices,'MarkerSize',5,'MarkerFaceColor','w','MarkerEdgeColor',curve_colors(curve_index,:));
end
if mark_initial_point
    initial_x = x_history{1}(1);
    initial_y = y_history{1}(1);
    plot(axes_handle,initial_x,initial_y,'o','Color','k','MarkerFaceColor','k','MarkerSize',5,'LineWidth',1);
end
xlabel(axes_handle,x_label,'Interpreter','latex','FontSize',24);
ylabel(axes_handle,y_label,'Interpreter','latex','FontSize',24);
xlim(axes_handle,[0,x_upper]);
if ~isfinite(y_upper)
    maximum_y = y_lower;
    for curve_index = 1:numel(y_history)
        current_x = x_history{curve_index};
        current_y = y_history{curve_index};
        visible_values = current_y(isfinite(current_x) & isfinite(current_y) & current_x<=x_upper & current_y>=y_lower);
        if ~isempty(visible_values)
            maximum_y = max(maximum_y,max(visible_values));
        end
    end
    y_upper = 10^ceil(log10(maximum_y));
end
y_upper = max(y_upper,10*y_lower);
ylim(axes_handle,[y_lower,y_upper]);
end

function export_combined_figure(figure_handle,png_file,pdf_file)
drawnow;
exportgraphics(figure_handle,png_file,'Resolution',600,'BackgroundColor','white');
exportgraphics(figure_handle,pdf_file,'ContentType','vector','BackgroundColor','white');
close(figure_handle);
end
