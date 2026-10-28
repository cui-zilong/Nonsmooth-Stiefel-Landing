function setup_paths
%SETUP_PATHS Add the algorithm and utility directories to the MATLAB path.

repository_root = fileparts(mfilename('fullpath'));
addpath(fullfile(repository_root,'algorithms'),'-begin');
addpath(fullfile(repository_root,'algorithms','baselines'),'-begin');
addpath(fullfile(repository_root,'utilities'),'-begin');
end
