%% MATLAB FOOTBALL PREDICTION SYSTEM - START HERE
% Complete implementation equivalent to Python/TensorFlow version

% ============================================================
% 📍 WHERE TO START
% ============================================================

% 1. FIRST TIME? Read MATLAB_IMPLEMENTATION_SUMMARY.md
%    → 5 min overview of what was created

% 2. QUICK START? Run this:
%    → Run football_prediction_main.m directly

% 3. WANT EXAMPLES? Run:
%    → examples_and_comparison.m (5 different configurations)

% 4. NEED DOCUMENTATION? Open:
%    → MATLAB_VERSION_README.md (100+ lines of detailed explanation)

% 5. COMING FROM PYTHON? Check:
%    → MIGRATION_GUIDE.m (Python to MATLAB patterns)

% 6. QUICK LOOKUP? See:
%    → QUICK_REFERENCE.m (Configuration templates)

% ============================================================
% 📂 FILE ORGANIZATION
% ============================================================

% MAIN ENTRY POINT:
%   football_prediction_main.m         - ⭐ Start here!
%
% CORE MODULES:
%   load_and_prepare_data.m            - Load CSV, preprocess
%   buildFNN.m                         - Feedforward network
%   buildCFNN.m                        - Cascade network
%   predict.m                          - Forward pass
%   trainNetwork.m                     - Training + optimizers
%   computeLoss.m                      - Loss functions
%
% DOCUMENTATION:
%   MATLAB_IMPLEMENTATION_SUMMARY.md   - Overview (START HERE!)
%   MATLAB_VERSION_README.md           - Full documentation
%   MIGRATION_GUIDE.m                  - Python equivalents
%   QUICK_REFERENCE.m                  - Quick lookup
%
% EXAMPLES:
%   examples_and_comparison.m          - 5 runnable examples
%   INDEX_START_HERE.m                 - This file!

% ============================================================
% 🎯 TYPICAL USAGE PATTERNS
% ============================================================

%% PATTERN 1: Default Training
%{
football_prediction_main
%}

%% PATTERN 2: Custom Configuration
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
config.dropout = [0.0, 0.0];

config.optimizer = 'Adam';
config.learning_rate = 5e-4;
config.loss_fn = 'mse';
config.epochs = 100;
config.batch_size = 32;
config.seed = 42;

% Load data
[X_train, y_train, X_val, y_val, features, target, y_mean, y_std] = ...
    load_and_prepare_data(config);

% Build network
net = buildFNN(size(X_train, 2), config.layers, config.activations, config.dropout);

% Train
[net, history] = trainNetwork(net, X_train, y_train, X_val, y_val, config);

% Evaluate
y_pred = predict(net, X_val);
val_loss = computeLoss(y_val, y_pred, config.loss_fn);
val_mae = mean(abs(y_val - y_pred));

fprintf('Validation Loss: %.6f, MAE: %.4f\n', val_loss, val_mae);

% Plot
figure;
plot(history.train_loss, 'b-', 'LineWidth', 2); hold on;
plot(history.val_loss, 'r-', 'LineWidth', 2);
xlabel('Epoch'); ylabel('Loss');
legend('Training', 'Validation');
grid on; title('Training Progress');
%}

%% PATTERN 3: Model Comparison
%{
clear; clc;

config = struct();
config.data_file = 'ftball.csv';
config.target = 'total_goals';
config.val_fraction = 0.2;
config.standardize_X = true;
config.epochs = 50;
config.batch_size = 32;
config.loss_fn = 'mse';

[X_train, y_train, X_val, y_val, ~, ~, ~, ~] = load_and_prepare_data(config);

models = struct();
models(1).name = 'FNN-Small';
models(1).config = config;
models(1).config.model_type = 'FNN';
models(1).config.layers = [16, 8];
models(1).config.activations = {'relu', 'relu'};
models(1).config.dropout = [0, 0];

models(2).name = 'FNN-Large';
models(2).config = config;
models(2).config.model_type = 'FNN';
models(2).config.layers = [64, 32, 16];
models(2).config.activations = {'relu', 'relu', 'relu'};
models(2).config.dropout = [0, 0, 0];

models(3).name = 'CFNN';
models(3).config = config;
models(3).config.model_type = 'CFNN';
models(3).config.layers = [32, 16];
models(3).config.activations = {'relu', 'relu'};
models(3).config.dropout = [0, 0];

results = [];
for i = 1:length(models)
    cfg = models(i).config;
    
    if strcmp(cfg.model_type, 'FNN')
        net = buildFNN(size(X_train, 2), cfg.layers, cfg.activations, cfg.dropout);
    else
        net = buildCFNN(size(X_train, 2), cfg.layers, cfg.activations, cfg.dropout);
    end
    
    [net, hist] = trainNetwork(net, X_train, y_train, X_val, y_val, cfg);
    y_pred = predict(net, X_val);
    final_loss = computeLoss(y_val, y_pred, cfg.loss_fn);
    final_mae = mean(abs(y_val - y_pred));
    
    results = [results; struct('model', models(i).name, 'loss', final_loss, 'mae', final_mae)];
    
    fprintf('%s: Loss=%.6f, MAE=%.4f\n', models(i).name, final_loss, final_mae);
end
%}

%% PATTERN 4: Optimizer Comparison
%{
% Compare different optimizers on same architecture

clear; clc;

config = struct();
config.data_file = 'ftball.csv';
config.target = 'total_goals';
config.val_fraction = 0.2;
config.standardize_X = true;
config.model_type = 'FNN';
config.layers = [32, 16];
config.activations = {'relu', 'relu'};
config.dropout = [0, 0];
config.loss_fn = 'mse';
config.epochs = 50;
config.batch_size = 32;
config.learning_rate = 5e-4;

[X_train, y_train, X_val, y_val, ~, ~, ~, ~] = load_and_prepare_data(config);

optimizers = {'Adam', 'SGD', 'RMSprop', 'Adagrad'};
all_histories = {};

for i = 1:length(optimizers)
    config.optimizer = optimizers{i};
    net = buildFNN(size(X_train, 2), config.layers, config.activations, config.dropout);
    [net, hist] = trainNetwork(net, X_train, y_train, X_val, y_val, config);
    all_histories{i} = hist;
end

% Plot comparison
figure('Position', [100, 100, 1200, 400]);
for i = 1:length(optimizers)
    subplot(2, 2, i);
    plot(all_histories{i}.train_loss, 'b-'); hold on;
    plot(all_histories{i}.val_loss, 'r-');
    xlabel('Epoch'); ylabel('Loss');
    title(['Optimizer: ' optimizers{i}]);
    legend('Train', 'Val'); grid on;
end
%}

%% PATTERN 5: Hyperparameter Tuning
%{
% Simple grid search

clear; clc;

config = struct();
config.data_file = 'ftball.csv';
config.target = 'total_goals';
config.val_fraction = 0.2;
config.standardize_X = true;
config.model_type = 'FNN';
config.loss_fn = 'mse';
config.epochs = 30;
config.batch_size = 32;
config.seed = 42;

[X_train, y_train, X_val, y_val, ~, ~, ~, ~] = load_and_prepare_data(config);

learning_rates = [1e-4, 5e-4, 1e-3];
layer_configs = {[16, 8], [32, 16], [64, 32, 16]};

best_loss = inf;
best_config = struct();

for lr_idx = 1:length(learning_rates)
    for layer_idx = 1:length(layer_configs)
        config.learning_rate = learning_rates(lr_idx);
        config.layers = layer_configs{layer_idx};
        config.activations = repmat({'relu'}, 1, length(config.layers));
        config.dropout = zeros(1, length(config.layers));
        
        net = buildFNN(size(X_train, 2), config.layers, config.activations, config.dropout);
        [net, ~] = trainNetwork(net, X_train, y_train, X_val, y_val, config);
        
        y_pred = predict(net, X_val);
        loss = computeLoss(y_val, y_pred, config.loss_fn);
        
        fprintf('LR=%.0e, Layers=%s: Loss=%.6f\n', config.learning_rate, mat2str(config.layers), loss);
        
        if loss < best_loss
            best_loss = loss;
            best_config = config;
        end
    end
end

fprintf('\nBest Configuration: LR=%.0e, Layers=%s, Loss=%.6f\n', ...
    best_config.learning_rate, mat2str(best_config.layers), best_loss);
%}

% ============================================================
% 📚 DOCUMENTATION INDEX
% ============================================================

% START HERE:
%   → MATLAB_IMPLEMENTATION_SUMMARY.md
%     (Overview of what was created)
%
% FOR USAGE:
%   → MATLAB_VERSION_README.md
%     (Complete feature documentation)
%
% FOR EXAMPLES:
%   → examples_and_comparison.m
%     (5 different configurations)
%
% FOR DEBUGGING:
%   → QUICK_REFERENCE.m
%     (Troubleshooting and tips)
%
% FOR CONVERSION:
%   → MIGRATION_GUIDE.m
%     (Python to MATLAB patterns)

% ============================================================
% 🎓 KEY CONCEPTS
% ============================================================

% 1. NEURAL NETWORK:
%    Mathematical model with layers of interconnected neurons
%    FNN: Linear connections to next layer
%    CFNN: Cascade connections from all previous layers

% 2. FORWARD PASS:
%    Compute prediction: Z = X * W + b, A = activation(Z)
%    See predict.m

% 3. TRAINING:
%    1. Forward pass: compute predictions
%    2. Compute loss: |y_true - y_pred|²
%    3. Compute gradients: ∂loss/∂weights
%    4. Update weights: W -= learning_rate * gradient
%    See trainNetwork.m

% 4. OPTIMIZERS:
%    Adam: Uses momentum and adaptive learning rates
%    SGD: Simple gradient descent with optional momentum
%    RMSprop: Adaptive learning rates using squared gradients
%    Adagrad: Accumulates squared gradients

% 5. HYPERPARAMETERS:
%    learning_rate: Step size for weight updates (5e-4)
%    batch_size: Samples per gradient update (32)
%    epochs: Full passes through training data (100)
%    dropout: Fraction of neurons to drop during training (0.1)
%    layers: Number of units in each hidden layer ([32, 16])

% ============================================================
% 🔧 COMMON COMMANDS
% ============================================================

% Run main script:
% >> football_prediction_main

% Run examples (5 different configurations):
% >> examples_and_comparison

% Train custom model:
% >> config = struct(); config.layers = [64, 32]; 
% >> [X_train, y_train, X_val, y_val, ~, ~, ~, ~] = load_and_prepare_data(config);
% >> net = buildFNN(size(X_train,2), config.layers, config.activations, config.dropout);
% >> [net, hist] = trainNetwork(net, X_train, y_train, X_val, y_val, config);

% Make predictions:
% >> y_pred = predict(net, X_new);

% Evaluate:
% >> loss = computeLoss(y_true, y_pred, 'mse');

% Plot training curves:
% >> plot(history.train_loss); hold on; plot(history.val_loss); legend('Train', 'Val');

% ============================================================
% ✨ FEATURES
% ============================================================

% ✓ Feedforward Neural Networks (FNN)
% ✓ Cascade Feedforward Neural Networks (CFNN)
% ✓ 7 Activation functions (ReLU, Tanh, Sigmoid, ELU, SELU, GELU, Linear)
% ✓ 4 Optimizers (Adam, SGD, RMSprop, Adagrad)
% ✓ 3 Loss functions (MSE, MAE, Huber)
% ✓ Data preprocessing (normalization, train/val split)
% ✓ Various regularization (dropout)
% ✓ Performance metrics (MAE, RMSE, Loss)
% ✓ Visualization (training curves, predictions, residuals)
% ✓ 5 Complete examples
% ✓ Comprehensive documentation

% ============================================================
% 🚀 QUICK START PATH
% ============================================================

% 1. Read MATLAB_IMPLEMENTATION_SUMMARY.md (5 min)
% 2. Run: football_prediction_main (2 min)
% 3. Explore: examples_and_comparison (5 min)
% 4. Customize: Edit config and rerun
% 5. Study: Check MIGRATION_GUIDE.m for Python patterns

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════════╗\n');
fprintf('║  MATLAB Football Prediction System - Ready to Use!             ║\n');
fprintf('║                                                                ║\n');
fprintf('║  📂 START HERE:                                                ║\n');
fprintf('║     1. Read: MATLAB_IMPLEMENTATION_SUMMARY.md                  ║\n');
fprintf('║     2. Run:  football_prediction_main                          ║\n');
fprintf('║     3. Try:  examples_and_comparison                           ║\n');
fprintf('║                                                                ║\n');
fprintf('║  📚 DOCUMENTATION:                                             ║\n');
fprintf('║     • MATLAB_VERSION_README.md - Full docs                     ║\n');
fprintf('║     • MIGRATION_GUIDE.m - Python to MATLAB                    ║\n');
fprintf('║     • QUICK_REFERENCE.m - Quick lookup                         ║\n');
fprintf('║                                                                ║\n');
fprintf('║  🎯 5 RUNNABLE EXAMPLES IN: examples_and_comparison           ║\n');
fprintf('╚════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');
