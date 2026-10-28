clear;
clc;
close all;

% Paper reproduction: Section 6.3, Table 4 (structured synthetic SPCA).

demo_directory = fileparts(mfilename('fullpath'));
repository_root = fileparts(demo_directory);
addpath(repository_root,'-begin');
setup_paths;
clear ManPGAda IPGLanding PGLanding RADMM SOC LSALM SLPG RieSub;

n_values = [200;300;400;500];
p_values = [100;150;200;250];
m = 1000;
gamma = 0.5;
seed = 0;

%% User-adjustable timing setting
number_of_repetitions = 10; % Change this value to control the number of timed runs.

settings = struct();
settings.tolerance = 1e-4;
settings.distance_threshold = 0.1;
settings.maximum_iterations = 30000;
settings.number_of_repetitions = number_of_repetitions;
settings.sparsity_tolerance = 1e-5;
settings.warm_iterations = 250;
settings.reference_tolerance = 1e-6;
settings.reference_inner_tolerance = 1e-12;
settings.reference_maximum_iterations = 100000;
settings.lsalm_profile = "synthetic";

ResultCells = cell(numel(n_values),1);
for case_index = 1:numel(n_values)
    n = n_values(case_index);
    p = p_values(case_index);
    fprintf('\n============================================================\n');
    fprintf('Structured synthetic SPCA: n=%d, m=%d, p=%d, gamma=%g, seed=%d\n',n,m,p,gamma,seed);
    fprintf('============================================================\n');
    [L,X_initial] = Gen_SPCA_Synthetic(n,p,m,seed);
    case_results = run_comparison_case_local(L,X_initial,gamma,p,settings);
    rows = height(case_results);
    case_results = addvars(case_results,repmat(n,rows,1),repmat(m,rows,1),repmat(p,rows,1),repmat(gamma,rows,1),zeros(rows,1),'Before',1,'NewVariableNames',{'n','m','p','gamma','Seed'});
    ResultCells{case_index} = case_results;
end

Results = vertcat(ResultCells{:});
disp(Results);
output_file = fullfile(demo_directory,'SPCA_complete_comparison_synthetic.csv');
writetable(Results,output_file);
fprintf('Saved %s\n',output_file);


function Results = run_comparison_case_local(L,X_initial,gamma,p,settings)
warm_option = struct();
warm_option.phi_init = X_initial;
warm_option.gamma = gamma;
warm_option.maxiter = settings.warm_iterations;
warm_option.tol = settings.tolerance;
warm_option.eta0 = 1;
warm_option.zero_tol = 0;
warm_option.stepsize_rule = 'power_three_quarters';
warm_option.stepsize_beta = 0.999;
warm_option.verbose = 0;
warm_option.comparison_mode = false;
X_warm = RieSub(L,warm_option);

reference_option = struct();
reference_option.phi_init = X_warm;
reference_option.gamma = gamma;
reference_option.maxiter = settings.reference_maximum_iterations;
reference_option.tol = settings.reference_tolerance;
reference_option.inner_iter = 100;
reference_option.inner_residual_tolerance = settings.reference_inner_tolerance;
reference_option.zero_tol = 0;
reference_option.verbose = 0;
reference_option.comparison_mode = false;
[X_reference,F_reference,~,reference_time,reference_iterations,reference_flag,~,~,reference_history] = ManPGAda(L,reference_option);
reference_stationarity = reference_history.stationarity;
reference_orthogonality = norm(X_reference'*X_reference-eye(p),'fro');
if reference_flag~=1 || reference_stationarity>settings.reference_tolerance || reference_orthogonality>1e-9
    error('ManPG-Ada failed to generate the requested reference solution.');
end
fprintf('Reference: F=% .10f, iter=%d, time=%.5f, stationarity=%.3e, orthogonality=%.3e\n',F_reference,reference_iterations,reference_time,reference_stationarity,reference_orthogonality);

method_names = ["ManPG-Ada";"IPGLanding";"PGLanding";"RADMM";"SOC";"LSALM";"SLPG"];
number_of_methods = numel(method_names);
number_of_rows = number_of_methods+1;
Method = strings(number_of_rows,1);
Objective = nan(number_of_rows,1);
Sparsity = nan(number_of_rows,1);
Iterations = nan(number_of_rows,1);
MeanTimeSeconds = nan(number_of_rows,1);
MeanTimePerIterationSeconds = nan(number_of_rows,1);
Orthogonality = nan(number_of_rows,1);
Flag = nan(number_of_rows,1);
AcceptedSolution = false(number_of_rows,1);
Status = strings(number_of_rows,1);

Method(1) = "Reference";
Objective(1) = F_reference;
Sparsity(1) = nnz(abs(X_reference)<=settings.sparsity_tolerance)/numel(X_reference);
Orthogonality(1) = reference_orthogonality;
Flag(1) = reference_flag;
AcceptedSolution(1) = true;
Status(1) = "reference";

screening_results = cell(number_of_methods,1);
for method_index = 1:number_of_methods
    method = method_names(method_index);
    fprintf('Screening %s\n',method);
    result = run_method_local(method,L,X_warm,F_reference,gamma,p,settings);
    objective_gap = result.Objective-F_reference;
    distance = norm(result.X-X_reference,'fro');
    accepted = result.Flag==1 && objective_gap<=settings.tolerance && result.Orthogonality<=settings.tolerance && distance<settings.distance_threshold;
    result.Accepted = accepted;
    screening_results{method_index} = result;
    row = method_index+1;
    Method(row) = method;
    Objective(row) = result.Objective;
    Sparsity(row) = nnz(abs(result.X)<=settings.sparsity_tolerance)/numel(result.X);
    Iterations(row) = result.Iterations;
    Orthogonality(row) = result.Orthogonality;
    Flag(row) = result.Flag;
    AcceptedSolution(row) = accepted;
    if result.Flag==0
        Status(row) = "iteration_limit";
    elseif accepted
        Status(row) = "accepted";
    else
        Status(row) = "postcheck_failed";
    end
    fprintf('%-12s iter=%6d F=% .9f orth=%.3e distance=%.3e accepted=%d\n',method,result.Iterations,result.Objective,result.Orthogonality,distance,accepted);
end

for method_index = 1:number_of_methods
    result = screening_results{method_index};
    if ~result.Accepted || result.Iterations>=settings.maximum_iterations
        continue
    end
    samples = nan(settings.number_of_repetitions,1);
    for repetition = 1:settings.number_of_repetitions
        timed_result = run_method_local(method_names(method_index),L,X_warm,F_reference,gamma,p,settings);
        timed_distance = norm(timed_result.X-X_reference,'fro');
        timed_accepted = timed_result.Flag==1 && timed_result.Objective-F_reference<=settings.tolerance && timed_result.Orthogonality<=settings.tolerance && timed_distance<settings.distance_threshold;
        if ~timed_accepted || timed_result.Iterations~=result.Iterations
            error('A repeated run of %s did not reproduce the screening result.',method_names(method_index));
        end
        samples(repetition) = timed_result.TimeSeconds;
    end
    row = method_index+1;
    MeanTimeSeconds(row) = mean(samples);
    MeanTimePerIterationSeconds(row) = MeanTimeSeconds(row)/max(Iterations(row),1);
end

Results = table(Method,Objective,Sparsity,Iterations,MeanTimeSeconds,MeanTimePerIterationSeconds,Orthogonality,Flag,AcceptedSolution,Status);
end

function result = run_method_local(method,L,X_warm,F_reference,gamma,p,settings)
option = struct();
option.phi_init = X_warm;
option.gamma = gamma;
option.maxiter = settings.maximum_iterations;
option.tol = settings.tolerance;
option.zero_tol = 0;
option.verbose = 0;
option.comparison_mode = true;
option.reference_objective = F_reference;

if method=="ManPG-Ada"
    option.inner_iter = 100;
    [X,F,~,elapsed,iterations,flag] = ManPGAda(L,option);
    orthogonality = norm(X'*X-eye(p),'fro');
elseif method=="IPGLanding"
    option.epsilon = 0.5;
    option.lambda = 5/12;
    option.dual_theta = 1.95;
    option.initial_primal_stepsize = 1.20;
    option.backtrack_factor = 0.4;
    option.nonmonotone_memory = 0.95;
    option.merit_rho = 1e-3;
    option.merit_sigma_factor = 1.1;
    option.merit_decrease_fraction = 1e-3;
    [X,F,orthogonality,~,elapsed,iterations,flag] = IPGLanding(L,option);
elseif method=="PGLanding"
    option.epsilon = 0.5;
    option.lambda = 0.5;
    option.fixed_t_factor = 1;
    option.ssn_maxiter = 20;
    option.inner_tol_min = 1e-12;
    option.inner_tol_max = 1e-7;
    option.inner_tol_forcing = 1e-5;
    option.matrix_free_tolerance_factor = 0.1;
    option.matrix_free_tolerance_min = 1e-12;
    option.matrix_free_tolerance_max = 1e-7;
    option.inner_failure_tightening = 0.1;
    [X,F,orthogonality,~,elapsed,iterations,flag] = PGLanding(L,option);
elseif method=="RADMM"
    [X,F,orthogonality,~,elapsed,iterations,flag] = RADMM(L,option);
elseif method=="SOC"
    [X,F,~,elapsed,iterations,flag] = SOC(L,option);
    orthogonality = norm(X'*X-eye(p),'fro');
elseif method=="SLPG"
    option.post_process = false;
    option.inner_maxiter = 5;
    option.inner_constant = 1000;
    option.initial_scale = 0.01;
    option.initial_steps = 5;
    option.record_history = false;
    option.bb_spectral_cap = 2.5;
    [X,F,orthogonality,~,elapsed,iterations,flag] = SLPG(L,option);
else
    option.scale_with_L = false;
    option.r = 15;
    option.beta = 0.5;
    option.lambda_scale = 1;
    option.dual_epsilon = 1e-10;
    option.RX = 10;
    option.RY = 1000;
    if settings.lsalm_profile=="random"
        option.rho = 10;
        option.alpha = round(0.07*sqrt(size(X_warm,1)*p));
    elseif settings.lsalm_profile=="synthetic"
        option.rho = 10;
        option.r = 25;
        option.alpha = round(0.07*sqrt(size(X_warm,1)*p));
    else
        option.rho = 100;
        option.alpha = 600;
    end
    [X,F,orthogonality,~,elapsed,iterations,flag] = LSALM(L,option);
end

result = struct('X',X,'Objective',F,'Orthogonality',orthogonality,'TimeSeconds',elapsed,'Iterations',iterations,'Flag',flag);
end

