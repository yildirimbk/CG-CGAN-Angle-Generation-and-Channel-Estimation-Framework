function [] = rayt_output_creator(a,b)
%Takes file names as input

real_rt_outputs = load(a);
real_rt_outputs = real_rt_outputs.real_rt_outputs_all_test;
generated_rt_outputs = load(b);
generated_rt_outputs = generated_rt_outputs.generated_rt_outputs_all_test;

size(real_rt_outputs)
size(generated_rt_outputs)

field_names= {'num_paths',...
               'DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta',...
               'phase', 'ToA', 'power'};

col_mapping = {1, 2:5, 6:9, 10:13, 14:17, 18:21, 22:25, 26:29};

data_real = struct([]);
data_generated = struct([]);

% for i=1:size(matrix,1)
% 
%     for j = 1:length(field_names)
% 
%         field_name = field_names{j};
%         cols = col_mapping{j};
%         data_real(i).(field_name) = single(real_rt_outputs(i, cols));  
%         data_generated(i).(field_name) = single(generated_rt_outputs(i, cols));  
% 
%         % % Ensure it's stored as a single-element array
%         if size(data_real(i).(field_name))==1
%             data_real(i).(field_name) = reshape(data_real(i).(field_name), [1, 1]);
%             data_generated(i).(field_name) = reshape(data_generated(i).(field_name), [1, 1]);
%         end
% 
%     end
% 
% end

for i = 1:size(real_rt_outputs,1)

    % Extract num_paths for the current row
    num_paths = real_rt_outputs(i, 1);  % First column represents num_paths
    num_keep = min(num_paths, 4);  % Ensure we don't exceed 4 (max columns available)

    for j = 1:length(field_names)
        field_name = field_names{j};  % Get current field name
        cols = col_mapping{j};        % Get corresponding columns

        % Extract and trim the values to match num_paths
        if j == 1  % 'num_paths' field (single value)
            data_real(i).(field_name) = single(num_paths);
            data_generated(i).(field_name) = single(num_paths);
        else  % Other fields (arrays)
            data_real(i).(field_name) = single(real_rt_outputs(i, cols(1:num_keep)));
            data_generated(i).(field_name) = single(generated_rt_outputs(i, cols(1:num_keep)));
        end
 % Ensure it's stored as a single-element array
        if size(data_real(i).(field_name))==1
            data_real(i).(field_name) = reshape(data_real(i).(field_name), [1, 1]);
            data_generated(i).(field_name) = reshape(data_generated(i).(field_name), [1, 1]);
        end
    end
end



save('real_rt_outputs_4path_after_matlab.mat', 'data_real');
save('generated_rt_outputs_after_matlab.mat', 'data_generated');
end

