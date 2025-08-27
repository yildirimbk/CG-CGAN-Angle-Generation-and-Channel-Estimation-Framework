function [] = see_dataset_distributions_function(training_dataset, test_dataset, value, num_bins, only_comb)

%Load Datasets
training_dataset_loaded_o = load(training_dataset);
test_dataset_loaded_o = load(test_dataset);

if isstruct(training_dataset_loaded_o)==1

    fieldNames = fieldnames(training_dataset_loaded_o);
    name = string(fieldNames);
    data = training_dataset_loaded_o.(name{1});

    if isstruct(data)==0
        
        training_dataset_loaded = data;

    else

        data2 = struct2cell(data);
        data3 = cell2mat(data2);
        training_dataset_loaded = data3;

    end
   
else
    training_dataset_loaded = training_dataset_loaded_o;
end

if isstruct(test_dataset_loaded_o)==1
    fieldNames = fieldnames(test_dataset_loaded_o);
    name = string(fieldNames);
    data = test_dataset_loaded_o.(name{1});

    if isstruct(data)==0
        
        test_dataset_loaded = data;

    else
        data2 = struct2cell(data);
        data3 = cell2mat(data2);
        test_dataset_loaded = data3;
    end
    
else
    test_dataset_loaded = test_dataset_loaded_o;

end

% Combine Datasets

combined_dataset = [training_dataset_loaded; test_dataset_loaded];

for i=1:size(value,2)

    value_loaded = value(i);

    % Extract the 7th column for each dataset
    training_column = training_dataset_loaded(:, value_loaded);
    test_column = test_dataset_loaded(:, value_loaded);
    combined_column = combined_dataset(:, value_loaded);

    if only_comb==0
        % Plot the distributions
        figure;
    
        % Training Dataset Distribution
        subplot(3, 1, 1);
        histogram(training_column, num_bins); % Adjust number of bins (e.g., 50) as needed
        title(['Distribution of Column ' , num2str(value_loaded) , ' Training Dataset']);
        xlabel('Values');
        ylabel('Frequency');
        grid on;
    
        % Test Dataset Distribution
        subplot(3, 1, 2);
        histogram(test_column, num_bins); % Adjust number of bins as needed
        title(['Distribution of Column ', num2str(value_loaded) , ' Test Dataset']);
        xlabel('Values');
        ylabel('Frequency');
        grid on;
    
        % Combined Dataset Distribution
        subplot(3, 1, 3);
        histogram(combined_column, num_bins); % Adjust number of bins as needed
        title(['Distribution of Column ', num2str(value_loaded) , ' Combined Dataset']);
    
        xlabel('Values');
        ylabel('Frequency');
        grid on;
    
        % Enhance readability
        sgtitle(['Comparative Distributions of Column ' , num2str(value_loaded)]);
    else
        figure;
        histogram(combined_column, num_bins); % Adjust number of bins as needed
        title(['Distribution of Column ', num2str(value_loaded) , ' Combined Dataset']);

    end
end

end
%see_dataset_distributions_function('training_dataset_allBS_5_k1_nopath_removed.mat', 'test_dataset_allBS_5_k1_nopath_removed.mat', 53, 500, 0)
