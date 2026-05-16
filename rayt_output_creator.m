function [] = rayt_output_creator(real_input_path, generated_input_path, max_paths, output_dir)
% RAYT_OUTPUT_CREATOR  Convert flat ray-tracing output matrices into structs of
%   per-sample fields with names matching the DeepMIMO generator API.
%
%   real_input_path       : path to the .mat file containing real_rt_outputs_all_test
%   generated_input_path  : path to the .mat file containing generated_rt_outputs_all_test
%   max_paths             : max number of paths per sample (e.g., 4)
%   output_dir            : where to write the converted .mat files (e.g., 'outputs')
%
%   Writes two files into output_dir:
%       real_rt_outputs_after_matlab.mat       (struct array 'data_real')
%       generated_rt_outputs_after_matlab.mat  (struct array 'data_generated')
%
%   The input matrices must have shape (N, 1 + 7*max_paths) with columns:
%       col 1                  : num_paths
%       cols 2..max_paths+1    : DoD_phi
%       cols ...               : DoD_theta, DoA_phi, DoA_theta, phase, ToA, power
%   (Each subsequent block of max_paths columns is the next field.)

if nargin < 3, max_paths  = 4;        end
if nargin < 4, output_dir = 'outputs'; end

if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% Load
real_data = load(real_input_path);
real_rt_outputs = real_data.real_rt_outputs_all_test;

generated_data = load(generated_input_path);
generated_rt_outputs = generated_data.generated_rt_outputs_all_test;

fprintf('Real RT outputs:      %s\n',      mat2str(size(real_rt_outputs)));
fprintf('Generated RT outputs: %s\n',      mat2str(size(generated_rt_outputs)));

% Field names and column ranges (DeepMIMO expects these exact names)
field_names = {'num_paths', ...
               'DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta', ...
               'phase', 'ToA', 'power'};

% Column mapping derived from max_paths
col_mapping = cell(1, length(field_names));
col_mapping{1} = 1;
for j = 2:length(field_names)
    start_col = 2 + (j - 2) * max_paths;
    col_mapping{j} = start_col : start_col + max_paths - 1;
end

% Validate input width
expected_cols = 1 + (length(field_names) - 1) * max_paths;
if size(real_rt_outputs, 2) ~= expected_cols
    error('Real RT outputs has %d cols, expected %d for max_paths=%d', ...
        size(real_rt_outputs, 2), expected_cols, max_paths);
end
if size(generated_rt_outputs, 2) ~= expected_cols
    error('Generated RT outputs has %d cols, expected %d for max_paths=%d', ...
        size(generated_rt_outputs, 2), expected_cols, max_paths);
end

% Preallocate struct arrays
n_samples = size(real_rt_outputs, 1);
template = struct();
for j = 1:length(field_names), template.(field_names{j}) = []; end
data_real      = repmat(template, n_samples, 1);
data_generated = repmat(template, n_samples, 1);

% Populate
for i = 1:n_samples
    num_paths = real_rt_outputs(i, 1);
    num_keep  = min(num_paths, max_paths);

    for j = 1:length(field_names)
        field_name = field_names{j};
        if j == 1
            data_real(i).(field_name)      = single(num_paths);
            data_generated(i).(field_name) = single(num_paths);
        else
            cols = col_mapping{j};
            data_real(i).(field_name)      = single(real_rt_outputs(i,      cols(1:num_keep)));
            data_generated(i).(field_name) = single(generated_rt_outputs(i, cols(1:num_keep)));
        end
    end
end

% Save
save(fullfile(output_dir, 'real_rt_outputs_after_matlab.mat'),      'data_real',      '-v7.3');
save(fullfile(output_dir, 'generated_rt_outputs_after_matlab.mat'), 'data_generated', '-v7.3');

fprintf('Saved: %s\n', fullfile(output_dir, 'real_rt_outputs_after_matlab.mat'));
fprintf('Saved: %s\n', fullfile(output_dir, 'generated_rt_outputs_after_matlab.mat'));
end



% Example usage:
% see_dataset_distributions_function('real_rt_outputs_O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1.mat', ...
%                                    'generated_rt_outputs_O1_60_Lp4_BS18_rows1-2751_TX8x4_RX4x2_sc1.mat', ...
%                                    4, 'outputs')


