%% MATLAB Football Prediction - Quick Start Examples
% This file demonstrates how to use the neural network system
% with different configurations

% Clear workspace
clear; clc; close all;

%% EXAMPLE 1: Simple FNN with default configuration
fprintf('\n=== EXAMPLE 1: Basic FNN ===\n');

config1 = struct();
config1.data_file = 'ftball.csv';
config1.target = 'total_goals';
config1.val_fraction = 0.2;
config1.standardize_X = true;
config1.standardize_y = false;
config1.model_type = 'FNN';
config1.layers = [32, 16];
config1.activations = {'relu', 'relu'};
config1.dropout = [0.0, 0.0];
config1.optimizer = 'Adam';
config1.learning_rate = 5e-4;
config1.loss_fn = 'mse';
config1.epochs = 50;
config1.batch_size = 32;
config1.seed = 42;

try
    [X_train1, y_train1, X_val1, y_val1, features1, target1, y_mean1, y_std1] = ...
        load_and_prepare_data(config1);
    
    net1 = buildFNN(size(X_train1, 2), config1.layers, config1.activations, config1.dropout);
    [net1, hist1] = trainNetwork(net1, X_train1, y_train1, X_val1, y_val1, config1);
    
    y_pred1 = predict(net1, X_val1);
    loss1 = computeLoss(y_val1, y_pred1, config1.loss_fn);
    mae1 = mean(abs(y_val1 - y_pred1));
    
    fprintf('Example 1 Results:\n');
    fprintf('  Final Validation Loss: %.6f\n', loss1);
    fprintf('  Validation MAE: %.4f\n', mae1);
    
catch ME
    fprintf('Example 1 Error: %s\n', ME.message);
end

%% EXAMPLE 2: CFNN with RMSprop optimizer
fprintf('\n=== EXAMPLE 2: CFNN with RMSprop ===\n');

config2 = struct();
config2.data_file = 'ftball.csv';
config2.target = 'home_goals';
config2.val_fraction = 0.2;
config2.standardize_X = true;
config2.standardize_y = false;
config2.model_type = 'CFNN';
config2.layers = [32, 16];
config2.activations = {'relu', 'relu'};
config2.dropout = [0.1, 0.1];
config2.optimizer = 'RMSprop';
config2.learning_rate = 1e-3;
config2.loss_fn = 'mae';
config2.epochs = 50;
config2.batch_size = 16;
config2.seed = 42;

try
    [X_train2, y_train2, X_val2, y_val2, features2, target2, y_mean2, y_std2] = ...
        load_and_prepare_data(config2);
    
    net2 = buildCFNN(size(X_train2, 2), config2.layers, config2.activations, config2.dropout);
    [net2, hist2] = trainNetwork(net2, X_train2, y_train2, X_val2, y_val2, config2);
    
    y_pred2 = predict(net2, X_val2);
    loss2 = computeLoss(y_val2, y_pred2, config2.loss_fn);
    mae2 = mean(abs(y_val2 - y_pred2));
    
    fprintf('Example 2 Results:\n');
    fprintf('  Final Validation Loss: %.6f\n', loss2);
    fprintf('  Validation MAE: %.4f\n', mae2);
    
catch ME
    fprintf('Example 2 Error: %s\n', ME.message);
end

%% EXAMPLE 3: Multiple optimizers comparison
fprintf('\n=== EXAMPLE 3: Comparing Optimizers ===\n');

optimizers_to_test = {'Adam', 'SGD', 'RMSprop'};
results3 = table();

try
    [X_train3, y_train3, X_val3, y_val3, features3, target3, y_mean3, y_std3] = ...
        load_and_prepare_data(config1);
    
    for opt_idx = 1:length(optimizers_to_test)
        config3 = config1;
        config3.optimizer = optimizers_to_test{opt_idx};
        config3.epochs = 30;  % Fewer epochs for comparison
        
        net3 = buildFNN(size(X_train3, 2), config1.layers, config1.activations, config1.dropout);
        [net3, ~] = trainNetwork(net3, X_train3, y_train3, X_val3, y_val3, config3);
        
        y_pred3 = predict(net3, X_val3);
        loss3 = computeLoss(y_val3, y_pred3, config1.loss_fn);
        mae3 = mean(abs(y_val3 - y_pred3));
        
        results3 = [results3; ...
            table({optimizers_to_test{opt_idx}}, loss3, mae3, ...
                  'VariableNames', {'Optimizer', 'Val_Loss', 'Val_MAE'})];
    end
    
    fprintf('Optimizer Comparison:\n');
    disp(results3);
    
catch ME
    fprintf('Example 3 Error: %s\n', ME.message);
end

%% EXAMPLE 4: Custom architecture with different activation functions
fprintf('\n=== EXAMPLE 4: Custom Architecture ===\n');

config4 = struct();
config4.data_file = 'ftball.csv';
config4.target = 'total_goals';
config4.val_fraction = 0.2;
config4.standardize_X = true;
config4.standardize_y = false;
config4.model_type = 'FNN';
config4.layers = [128, 64, 32];  % Deeper network
config4.activations = {'relu', 'tanh', 'relu'};  % Mixed activations
config4.dropout = [0.2, 0.15, 0.1];  % Increasing dropout
config4.optimizer = 'Adam';
config4.learning_rate = 1e-4;
config4.loss_fn = 'huber';
config4.epochs = 50;
config4.batch_size = 32;
config4.seed = 42;

try
    [X_train4, y_train4, X_val4, y_val4, features4, target4, y_mean4, y_std4] = ...
        load_and_prepare_data(config4);
    
    net4 = buildFNN(size(X_train4, 2), config4.layers, config4.activations, config4.dropout);
    [net4, hist4] = trainNetwork(net4, X_train4, y_train4, X_val4, y_val4, config4);
    
    y_pred4 = predict(net4, X_val4);
    loss4 = computeLoss(y_val4, y_pred4, config4.loss_fn);
    mae4 = mean(abs(y_val4 - y_pred4));
    rmse4 = sqrt(mean((y_val4 - y_pred4).^2));
    
    fprintf('Example 4 Results (Deep Network with Mixed Activations):\n');
    fprintf('  Final Validation Loss: %.6f\n', loss4);
    fprintf('  Validation MAE: %.4f\n', mae4);
    fprintf('  Validation RMSE: %.4f\n', rmse4);
    
catch ME
    fprintf('Example 4 Error: %s\n', ME.message);
end

%% EXAMPLE 5: Create comparison plots
fprintf('\n=== EXAMPLE 5: Visualization ===\n');

% Create comparison figure if examples 1 and 2 succeeded
if exist('hist1', 'var') && exist('hist2', 'var')
    figure('Position', [100, 100, 1400, 500]);
    
    % Plot 1: FNN training history
    subplot(2, 3, 1);
    plot(hist1.train_loss, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(hist1.val_loss, 'r-', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Example 1: FNN - MSE Loss');
    legend('Train', 'Val');
    grid on;
    
    % Plot 2: CFNN training history
    subplot(2, 3, 2);
    plot(hist2.train_loss, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(hist2.val_loss, 'r-', 'LineWidth', 1.5);
    xlabel('Epoch'); ylabel('Loss');
    title('Example 2: CFNN - MAE Loss');
    legend('Train', 'Val');
    grid on;
    
    % Plot 3: FNN predictions
    if exist('y_pred1', 'var')
        subplot(2, 3, 3);
        scatter(y_val1, y_pred1, 30, 'filled', 'Alpha', 0.6);
        hold on;
        lims = [min(y_val1), max(y_val1)];
        plot(lims, lims, 'r--', 'LineWidth', 2);
        xlabel('Actual'); ylabel('Predicted');
        title('Example 1: FNN - Val Predictions');
        grid on;
    end
    
    % Plot 4: CFNN predictions
    if exist('y_pred2', 'var')
        subplot(2, 3, 4);
        scatter(y_val2, y_pred2, 30, 'filled', 'Alpha', 0.6);
        hold on;
        lims = [min(y_val2), max(y_val2)];
        plot(lims, lims, 'r--', 'LineWidth', 2);
        xlabel('Actual'); ylabel('Predicted');
        title('Example 2: CFNN - Val Predictions');
        grid on;
    end
    
    % Plot 5: FNN residuals
    if exist('y_pred1', 'var')
        subplot(2, 3, 5);
        residuals1 = y_val1 - y_pred1;
        histogram(residuals1, 20, 'FaceAlpha', 0.7);
        xlabel('Residuals'); ylabel('Frequency');
        title('Example 1: FNN - Residuals');
        grid on;
    end
    
    % Plot 6: CFNN residuals
    if exist('y_pred2', 'var')
        subplot(2, 3, 6);
        residuals2 = y_val2 - y_pred2;
        histogram(residuals2, 20, 'FaceAlpha', 0.7);
        xlabel('Residuals'); ylabel('Frequency');
        title('Example 2: CFNN - Residuals');
        grid on;
    end
    
    sgtitle('Football Prediction System - Examples Comparison');
    savefig('examples_comparison.fig');
    fprintf('Comparison plot saved as examples_comparison.fig\n');
end

fprintf('\n=== All Examples Completed ===\n');

%% Function to print network statistics
function print_network_stats(net, X_train, y_train, config)
    fprintf('Network Architecture:\n');
    fprintf('  Type: %s\n', net.type);
    fprintf('  Input Dimension: %d\n', net.input_dim);
    fprintf('  Output Dimension: %d\n', net.output_dim);
    
    if strcmp(net.type, 'FNN')
        total_params = 0;
        fprintf('  Layers:\n');
        for i = 1:length(net.layers) - 1
            layer = net.layers{i};
            n_params = numel(net.weights{i}) + numel(net.biases{i});
            total_params = total_params + n_params;
            fprintf('    Dense (%s): %d units, Activation: %s\n', ...
                layer.type, layer.units, layer.activation);
            fprintf('      Parameters: %d\n', n_params);
        end
        fprintf('  Total Parameters: %d\n', total_params);
    end
end
