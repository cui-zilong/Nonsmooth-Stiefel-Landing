function [L,X_init,info] = Gen_SPCA_Realworld(dataset_name,p,seed,option)
%GEN_SPCA_REALWORLD Colon data with centered, unit-variance features.

if ~strcmpi(dataset_name,'colon')
    error('Only the Colon data set is supported.');
end

generator_directory = fileparts(mfilename('fullpath'));
repository_root = fileparts(generator_directory);
data_file = fullfile(repository_root,'data','ColonX.csv');
if ~isfile(data_file)
    error('ColonX.csv is not distributed with this repository. Download it from https://github.com/alxiang/Colon-Cancer-Classification/blob/master/ColonX.csv and save it as %s.',data_file);
end
A = readmatrix(data_file,'NumHeaderLines',1);
A = A(~all(isnan(A),2),:);
if size(A,2)==2001
    A = A(:,2:end);
end
original_feature_count = size(A,2);
A = A(:,any(isfinite(A),1));
for column = 1:size(A,2)
    observed = isfinite(A(:,column));
    A(~observed,column) = mean(A(observed,column));
end
A = A(:,max(A,[],1)>min(A,[],1));

number_of_samples = size(A,1);
A = A-mean(A,1);
feature_scale = sqrt(sum(A.^2,1)/(number_of_samples-1));
A = A./feature_scale;
L = 2*(A'*A)/(number_of_samples-1);
L = 0.5*(L+L');

random_stream = RandStream('mt19937ar','Seed',seed);
random_matrix = randn(random_stream,size(L,1),p);
[X_init,R] = qr(random_matrix,0);
diagonal_sign = sign(diag(R));
diagonal_sign(diagonal_sign==0) = 1;
X_init = X_init*diag(diagonal_sign);

info.name = lower(string(dataset_name));
info.number_of_samples = number_of_samples;
info.original_feature_count = original_feature_count;
info.retained_feature_count = size(A,2);
info.L_size = size(L);
info.initial_orthogonality = norm(X_init'*X_init-eye(p),'fro');

if option.verbose
    fprintf('Dataset=%s, samples=%d, features=%d, p=%d, initial orthogonality=%.3e\n',info.name,number_of_samples,size(L,1),p,info.initial_orthogonality);
end
end
