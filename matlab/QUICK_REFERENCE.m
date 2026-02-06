%% Quick Reference Card - MATLAB Football Prediction System
% Save this file or print for quick lookup

%% ============================================================
%% FILE ORGANIZATION
%% ============================================================

% football_prediction_main.m      - Main entry point, run this!
% load_and_prepare_data.m         - Load CSV and preprocess
% buildFNN.m                      - Build feedforward network
% buildCFNN.m                     - Build cascade network
% predict.m                       - Forward pass (inference)
% trainNetwork.m                  - Training loop + optimizers
% computeLoss.m                   - Loss functions (standalone)
% examples_and_comparison.m       - 5 complete examples
% MIGRATION_GUIDE.m               - Python to MATLAB guide
% MATLAB_VERSION_README.md        - Full documentation

%% ============================================================
%% QUICK START (Copy-Paste Template)
%% ============================================================

% TEMPLATE: Minimal working example
%{
clear; clc;

config = struct();
config.data_file = 'ftball.csv';
config.target = 'total_goals';
config.val_fraction = 0.2;
config.standardize_X = true;
config.standardize_y = false;
config.model_type = 'FNN';
config.layers = [32, 16];
config.activations = {'relu', 'relu'};
config.dropout = [0, 0];
config.optimizer = 'Adam';
config.learning_rate = 5e-4;
config.loss_fn = 'mse';
config.epochs = 50;
config.batch_size = 32;
config.seed = 42;

[X_train, y_train, X_val, y_val, ~, ~, ~, ~] = load_and_prepare_data(config);
net = buildFNN(size(X_train, 2), config.layers, config.activations, config.dropout);
[net, history] = trainNetwork(net, X_train, y_train, X_val, y_val, config);
y_pred = predict(net, X_val);
loss = computeLoss(y_val, y_pred, config.loss_fn);
fprintf('Validation Loss: %.6f\n', loss);
%}

%% ============================================================
%% CONFIGURATION CHEAT SHEET
%% ============================================================

% DATA SETTINGS
% config.data_file           - CSV filename (string)
% config.target              - Target column name (string)
% config.val_fraction        - Validation split (0.0-1.0)
% config.standardize_X       - Normalize features (true/false)
% config.standardize_y       - Normalize target (true/false)

% MODEL SETTINGS
% config.model_type          - 'FNN' or 'CFNN'
% config.layers              - [n1, n2, n3] units per layer
% config.activations         - {'relu', 'tanh', 'sigmoid',...}
% config.dropout             - [d1, d2, d3] rate per layer
%
% Supported activations: 'relu', 'tanh', 'sigmoid', 'elu', 'selu', 'gelu', 'linear'

% TRAINING SETTINGS
% config.optimizer           - 'Adam', 'SGD', 'RMSprop', 'Adagrad'
% config.learning_rate       - Learning rate (1e-6 to 1.0)
% config.loss_fn             - 'mse', 'mae', 'huber'
% config.epochs              - Number of training iterations
% config.batch_size          - Samples per batch
% config.seed                - Random seed for reproducibility

%% ============================================================
%% COMMON LEARNING RATES
%% ============================================================

% Start with these and adjust based on loss curves:
% 5e-4     - Default, works well for many problems
% 1e-3     - Larger steps, faster but less stable
% 1e-4     - Smaller steps, more stable but slower
% 1e-5     - Very conservative, for fine-tuning

%% ============================================================
%% ARCHITECTURE RECOMMENDATIONS
%% ============================================================

% SMALL DATASET (< 100 samples)
% config.layers = [16, 8];
% config.dropout = [0.2, 0.1];
% Fewer parameters to prevent overfitting

% MEDIUM DATASET (100-1000 samples)
% config.layers = [32, 16];
% config.dropout = [0.1, 0.0];
% Moderate complexity

% LARGE DATASET (> 1000 samples)
% config.layers = [64, 32, 16];
% config.dropout = [0.0, 0.0, 0.0];
% Can use deeper networks

% HIGH-DIMENSIONAL (100+ features)
% config.layers = [128, 64];
% config.activations = {'relu', 'relu'};
% Larger layers to preserve information

%% ============================================================
%% TROUBLESHOOTING
%% ============================================================

% ISSUE: Loss doesn't decrease
% FIX: Try smaller learning_rate (1e-5 to 1e-4)

% ISSUE: Validation loss worse than training (overfitting)
% FIX: Increase dropout, reduce layer sizes, use L2 regularization

% ISSUE: Training too slow
% FIX: Increase batch_size, reduce epochs for testing, use GPU

% ISSUE: NaN or Inf in predictions
% FIX: Check data normalization, reduce learning_rate, verify activations

% ISSUE: Model completely wrong
% FIX: Check data is loaded correctly, verify target variable name

%% ============================================================
%% KEY FUNCTIONS REFERENCE
%% ============================================================

% DATA LOADING & PREPROCESSING
% [X_train, y_train, X_val, y_val, feature_names, target, y_mean, y_std] = ...
%     load_and_prepare_data(config)

% BUILD MODELS
% net = buildFNN(input_dim, layers, activations, dropout)
% net = buildCFNN(input_dim, layers, activations, dropout)

% PREDICTION
% y_pred = predict(net, X)
% [y_pred, activations, z_vals] = predict(net, X)

% TRAINING
% [net, history] = trainNetwork(net, X_train, y_train, X_val, y_val, config)
%   Returns:
%   - net: trained network
%   - history.train_loss: training loss per epoch
%   - history.val_loss: validation loss per epoch
%   - history.train_mae: training MAE per epoch
%   - history.val_mae: validation MAE per epoch

% LOSS COMPUTATION
% loss = computeLoss(y_true, y_pred, 'mse')
%   Options: 'mse', 'mae', 'huber'

%% ============================================================
%% MONITORING TRAINING
%% ============================================================

% During training, watch for:
% 1. Loss should DECREASE (or at least not oscillate wildly)
% 2. Training and validation loss should stay CLOSE (not diverge)
% 3. MAE should be INTERPRETABLE in data units

% After training:
% 1. Plot predictions vs actuals: scatter(y_actual, y_pred)
% 2. Check residuals: histogram(y_actual - y_pred)
% 3. Compare to baseline: what if we just predicted mean?

%% ============================================================
%% PERFORMANCE METRICS
%% ============================================================

% After training, you can compute:
% 
% MAE (Mean Absolute Error) - avg error in data units
% mae = mean(abs(y_true - y_pred))
%
% RMSE (Root Mean Square Error) - penalizes large errors
% rmse = sqrt(mean((y_true - y_pred).^2))
%
% R² (Coefficient of determination) - % variance explained
% ss_res = sum((y_true - y_pred).^2)
% ss_tot = sum((y_true - mean(y_true)).^2)
% r2 = 1 - ss_res / ss_tot

%% ============================================================
%% ADVANCED: CUSTOM CONFIGURATIONS
%% ============================================================

% FNN with many layers
config.model_type = 'FNN';
config.layers = [256, 128, 64, 32, 16];
config.activations = {'relu', 'relu', 'relu', 'relu', 'relu'};
config.dropout = [0.3, 0.2, 0.1, 0.1, 0.0];

% CFNN (cascade connections)
config.model_type = 'CFNN';
config.layers = [32, 16];
config.activations = {'relu', 'relu'};
config.dropout = [0.1, 0.0];

% Conservative training
config.learning_rate = 1e-4;
config.batch_size = 64;
config.epochs = 200;
config.optimizer = 'SGD';

% Aggressive training
config.learning_rate = 1e-3;
config.batch_size = 16;
config.epochs = 50;
config.optimizer = 'Adam';

% Different loss functions
config.loss_fn = 'mse';    % Standard, penalizes large errors
config.loss_fn = 'mae';    % Robust to outliers
config.loss_fn = 'huber';  % Balanced

%% ============================================================
%% DEBUGGING PRINTS
%% ============================================================

% Add these to football_prediction_main.m or your script:

% Check data loading
fprintf('X_train shape: %d x %d\n', size(X_train, 1), size(X_train, 2));
fprintf('y_train shape: %d x %d\n', size(y_train, 1), size(y_train, 2));
fprintf('X_train mean: %.4f, std: %.4f\n', mean(X_train(:)), std(X_train(:)));
fprintf('y_train mean: %.4f, std: %.4f\n', mean(y_train), std(y_train));

% Check network structure
fprintf('Network type: %s\n', net.type);
fprintf('Number of parameters: %d\n', sum(cellfun(@numel, net.weights)));

% Check predictions
fprintf('y_pred min: %.4f, max: %.4f\n', min(y_pred), max(y_pred));
fprintf('y_actual min: %.4f, max: %.4f\n', min(y_val), max(y_val));

% Check training history
figure;
plot(history.train_loss, 'b-', 'LineWidth', 2);
hold on;
plot(history.val_loss, 'r-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
legend('Training', 'Validation');
grid on;
title('Training Progress');

%% ============================================================
%% EXPORTING AND SAVING
%% ============================================================

% Save trained network (MATLAB format)
save('my_trained_network.mat', 'net');

% Load trained network
load('my_trained_network.mat', 'net');
y_pred = predict(net, X_new);

% Export predictions to CSV
predictions_table = table(y_val, y_pred, y_val - y_pred, ...
    'VariableNames', {'Actual', 'Predicted', 'Residual'});
writetable(predictions_table, 'predictions.csv');

% Export training history to CSV
history_table = table(history.train_loss(:), history.val_loss(:), ...
    history.train_mae(:), history.val_mae(:), ...
    'VariableNames', {'Train_Loss', 'Val_Loss', 'Train_MAE', 'Val_MAE'});
writetable(history_table, 'training_history.csv');

%% ============================================================
%% HELPFUL MATLAB COMMANDS
%% ============================================================

% Basic matrix operations
A = randn(100, 50);          % Random matrix
b = zeros(50, 1);            % Zero vector
c = ones(100, 1);            % Ones vector
D = cat(2, A, b);            % Concatenate horizontally
E = [A; A];                  % Concatenate vertically

% Statistics
mean(data)                   % Average
std(data)                    % Standard deviation
min(data)                    % Minimum
max(data)                    % Maximum
sum(data)                    % Sum
numel(data)                  % Number of elements
size(data)                   % Dimensions

% Logical operations
idx = data > 0;              % Boolean mask
data(idx) = 0;               % Set where mask is true
data(~idx) = -1;             % Set where mask is false

% File I/O
T = readtable('file.csv');   % Read CSV table
writetable(T, 'out.csv');    % Write CSV table
save('file.mat', 'var');     % Save MAT file
load('file.mat', 'var');     % Load MAT file

% Visualization
plot(x, y);                  % Line plot
scatter(x, y);               % Scatter plot
histogram(x);                % Histogram
heatmap(Z);                  % Heatmap
figure;                      % New figure
subplot(2, 2, 1);            % Subplot
hold on;                     % Keep previous plots
grid on;                     % Grid lines
xlabel('Label');             % X-axis label
ylabel('Label');             % Y-axis label
title('Title');              % Title

%% ============================================================
%% CONTACT & SUPPORT
%% ============================================================

% For bugs, questions, or improvements:
% 1. Check MATLAB_VERSION_README.md for documentation
% 2. Review examples_and_comparison.m for usage patterns
% 3. Check MIGRATION_GUIDE.m for Python equivalents
% 4. Run disp(net) to see network structure
% 5. Use MATLAB profiler: profile on; ...; profile off; profview

fprintf('\n=== Quick Reference Card ===\n');
fprintf('Read this file for quick lookup: QUICK_REFERENCE.m\n');
fprintf('For full documentation: MATLAB_VERSION_README.md\n');
fprintf('For examples: examples_and_comparison.m\n');
fprintf('For Python transition: MIGRATION_GUIDE.m\n');
