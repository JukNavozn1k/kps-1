function [X_train, y_train, X_val, y_val, feature_names, target_name, y_mean, y_std] = ...
    load_and_prepare_data(config)
%LOAD_AND_PREPARE_DATA Load and preprocess football dataset
%
% Returns:
%   X_train, y_train: Training features and targets
%   X_val, y_val: Validation features and targets
%   feature_names: Cell array of feature names
%   target_name: Name of target variable
%   y_mean, y_std: Mean and std of target (if standardized)

    % Load CSV file
    opts = detectImportOptions(config.data_file);
    data_table = readtable(config.data_file, opts);
    
    % Extract target variable
    target_col = config.target;
    if ~ismember(target_col, data_table.Properties.VariableNames)
        error('Target column "%s" not found in data', target_col);
    end
    
    y = data_table.(target_col);
    y = table2array(y);
    
    % Extract features (all except target)
    feature_cols = setdiff(data_table.Properties.VariableNames, {target_col});
    
    % Filter numeric columns only
    feature_cols_numeric = {};
    for i = 1:length(feature_cols)
        col_data = data_table.(feature_cols{i});
        if isnumeric(col_data) || islogical(col_data)
            feature_cols_numeric{end+1} = feature_cols{i};
        end
    end
    
    feature_names = feature_cols_numeric;
    X = table2array(data_table(:, feature_names));
    
    % Convert to double
    X = double(X);
    y = double(y);
    
    % Remove rows with NaN
    valid_idx = ~any(isnan(X), 2) & ~isnan(y);
    X = X(valid_idx, :);
    y = y(valid_idx, :);
    
    % Standardize features
    if config.standardize_X
        X_mean = mean(X, 1);
        X_std = std(X, 1);
        X_std(X_std == 0) = 1;  % Avoid division by zero
        X = (X - X_mean) ./ X_std;
    end
    
    % Standardize target
    y_mean = [];
    y_std = [];
    if config.standardize_y
        y_mean = mean(y);
        y_std = std(y);
        if y_std == 0
            y_std = 1;
        end
        y = (y - y_mean) ./ y_std;
    end
    
    % Split into train/val
    n = length(y);
    n_val = round(n * config.val_fraction);
    n_train = n - n_val;
    
    % Random split
    rng(config.seed);
    idx = randperm(n);
    train_idx = idx(1:n_train);
    val_idx = idx(n_train+1:end);
    
    X_train = X(train_idx, :);
    y_train = y(train_idx);
    X_val = X(val_idx, :);
    y_val = y(val_idx);
    
    % Ensure column vectors
    y_train = y_train(:);
    y_val = y_val(:);
    
    target_name = target_col;
    
    fprintf('Data preparation complete:\n');
    fprintf('  Samples: %d total, %d train, %d val\n', n, n_train, n_val);
    fprintf('  Features: %d\n', size(X_train, 2));
end
