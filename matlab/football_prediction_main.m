%% Football Match Prediction System (MATLAB Version)
% This script demonstrates a complete pipeline for predicting football match results
% using neural networks (FNN and CFNN) with various optimization methods

clear; clc; close all;

%% Configuration
config = struct();
config.data_file = 'ftball.csv';
config.target = 'total_goals';  % 'home_goals', 'away_goals', 'total_goals', 'goal_diff'
config.val_fraction = 0.2;
config.standardize_X = true;
config.standardize_y = false;

% Model configuration
config.model_type = 'FNN';  % 'FNN' or 'CFNN'
config.layers = [32, 16];  % Number of units in each hidden layer
config.activations = {'relu', 'relu'};  % Activation functions
config.dropout = [0.0, 0.0];  % Dropout rates

% Training configuration
config.optimizer = 'Adam';  % 'Adam', 'SGD', 'RMSprop', 'Adagrad'
config.learning_rate = 5e-4;
config.loss_fn = 'mse';  % 'mse', 'mae', 'huber'
config.epochs = 100;
config.batch_size = 32;
config.seed = 42;

%% Load and preprocess data
fprintf('Loading data...\n');
[X_train, y_train, X_val, y_val, feature_names, target_name, y_mean, y_std] = ...
    load_and_prepare_data(config);

fprintf('Data loaded successfully!\n');
fprintf('Training set size: %d x %d\n', size(X_train));
fprintf('Validation set size: %d x %d\n', size(X_val));
fprintf('Features: %s\n', strjoin(feature_names, ', '));

%% Build neural network model
fprintf('\nBuilding %s model...\n', config.model_type);

if strcmp(config.model_type, 'FNN')
    net = buildFNN(size(X_train, 2), config.layers, config.activations, config.dropout);
elseif strcmp(config.model_type, 'CFNN')
    net = buildCFNN(size(X_train, 2), config.layers, config.activations, config.dropout);
else
    error('Unknown model type: %s', config.model_type);
end

fprintf('Model built successfully!\n');
fprintf('Network architecture:\n');
disp(net);

%% Train model
fprintf('\nTraining model...\n');

[net, history] = trainNetwork(net, X_train, y_train, X_val, y_val, config);

fprintf('Training completed!\n');

%% Evaluate model
fprintf('\nEvaluating model...\n');

y_train_pred = predict(net, X_train);
y_val_pred = predict(net, X_val);

train_loss = computeLoss(y_train, y_train_pred, config.loss_fn);
val_loss = computeLoss(y_val, y_val_pred, config.loss_fn);
train_mae = mean(abs(y_train - y_train_pred));
val_mae = mean(abs(y_val - y_val_pred));
train_rmse = sqrt(mean((y_train - y_train_pred).^2));
val_rmse = sqrt(mean((y_val - y_val_pred).^2));

fprintf('Training Loss: %.6f | MAE: %.4f | RMSE: %.4f\n', train_loss, train_mae, train_rmse);
fprintf('Validation Loss: %.6f | MAE: %.4f | RMSE: %.4f\n', val_loss, val_mae, val_rmse);

%% Visualize results
fprintf('\nGenerating visualizations...\n');

% Plot training curves
figure('Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
plot(history.train_loss, 'b-', 'LineWidth', 2);
hold on;
plot(history.val_loss, 'r-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
title(sprintf('Training Curves (%s)', config.loss_fn));
legend('Training Loss', 'Validation Loss');
grid on;

% Plot predictions vs actual
subplot(1, 3, 2);
scatter(y_val, y_val_pred, 30, 'filled', 'Alpha', 0.6);
hold on;
plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], 'r--', 'LineWidth', 2);
xlabel('Actual'); ylabel('Predicted');
title('Validation Predictions');
grid on;

% Plot residuals
subplot(1, 3, 3);
residuals = y_val - y_val_pred;
histogram(residuals, 30, 'FaceAlpha', 0.7);
xlabel('Residuals'); ylabel('Frequency');
title('Residuals Distribution');
grid on;

savefig('football_predictions_results.fig');
fprintf('Figures saved!\n');

%% Summary
fprintf('\n=== SUMMARY ===\n');
fprintf('Model Type: %s\n', config.model_type);
fprintf('Target Variable: %s\n', target_name);
fprintf('Optimizer: %s (LR: %.6f)\n', config.optimizer, config.learning_rate);
fprintf('Loss Function: %s\n', config.loss_fn);
fprintf('Validation Loss: %.6f\n', val_loss);
fprintf('Validation MAE: %.4f\n', val_mae);
fprintf('Validation RMSE: %.4f\n', val_rmse);
