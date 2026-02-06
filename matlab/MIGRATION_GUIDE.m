%% Migration Guide: Python to MATLAB
% This guide explains how the Python implementation was converted to MATLAB
% and provides patterns for extending the system

%% DATA LOADING AND PREPROCESSING

% ============ Python ============
% df = pd.read_csv("ftball.csv")
% X = df[[features]].values
% y = df[[target]].values
% X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
% X = (X - X.mean(axis=0)) / X.std(axis=0)

% ============ MATLAB ============
% See: load_and_prepare_data.m
% 1. Use readtable() to load CSV
% 2. Extract numeric columns with table2array()
% 3. Compute mean and std with mean() and std()
% 4. Use indexing to create train/val split

%% NEURAL NETWORK ARCHITECTURE

% ============ Python (TensorFlow/Keras) ============
% model = Sequential([
%     Dense(32, activation='relu', input_shape=(n_features,)),
%     Dropout(0.2),
%     Dense(16, activation='relu'),
%     Dense(1, activation='linear')
% ])

% ============ MATLAB ============
% See: buildFNN.m, buildCFNN.m
% Use structure arrays to represent network:
% net = struct();
% net.layers = {...};           % Layer descriptions
% net.weights = {...};          % Weight matrices
% net.biases = {...};           % Bias vectors
% net.activations = {...};      % Activation function names

%% FORWARD PASS (PREDICTION)

% ============ Python ============
% y_pred = model.predict(X)
% y_pred = model(X)  # TensorFlow eager execution

% ============ MATLAB ============
% See: predict.m
% for i = 1:n_layers
%     Z = A * W{i} + b{i}                    % Linear
%     A = apply_activation(Z, activation)   % Activation
% end

%% LOSS FUNCTIONS

% ============ Python ============
% mse_loss = tf.keras.losses.MeanSquaredError()
% mae_loss = tf.keras.losses.MeanAbsoluteError()
% mae = K.mean(K.abs(y_true - y_pred))

% ============ MATLAB ============
% See: computeLoss.m
% mse = mean((y_true - y_pred).^2)
% mae = mean(abs(y_true - y_pred))
% Loss computation is direct using MATLAB matrix operations

%% OPTIMIZERS

% ============ Python (Adam) ============
% optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
% The optimizer automatically computes gradients and updates weights

% ============ MATLAB (Adam) ============
% See: adamUpdate() in trainNetwork.m
% Manual implementation mimicking TensorFlow:
% opt.m = beta1 * opt.m + (1-beta1) * gradient       % First moment
% opt.v = beta2 * opt.v + (1-beta2) * gradient^2    % Second moment
% m_hat = opt.m / (1 - beta1^t)
% v_hat = opt.v / (1 - beta2^t)
% weights = weights - lr * m_hat / (sqrt(v_hat) + epsilon)

%% TRAINING LOOP

% ============ Python ============
% model.compile(optimizer=optimizer, loss=loss_fn)
% history = model.fit(X_train, y_train,
%                     validation_data=(X_val, y_val),
%                     epochs=100,
%                     batch_size=32)

% ============ MATLAB ============
% See: trainNetwork.m
% Manual implementation:
% for epoch = 1:n_epochs
%     Shuffle training data
%     for batch = 1:n_batches
%         y_pred = predict(net, X_batch)
%         loss = computeLoss(y_batch, y_pred)
%         [dW, db] = computeGradients(...)
%         [net, opt] = updateWeights(net, dW, db, opt)
%     end
%     Validate on full validation set
% end

%% ACTIVATION FUNCTIONS

% ============ Mapping ============
% Python (TensorFlow)           →  MATLAB
% tf.keras.activations.relu()   →  max(0, Z)
% tf.nn.tanh()                  →  tanh(Z)
% tf.nn.sigmoid()               →  1 / (1 + exp(-Z))
% tf.nn.elu()                   →  Custom ELU implementation
% tf.nn.gelu()                  →  Approximate GELU implementation

% ============ Python ============
% def custom_activation(x):
%     return tf.nn.relu(x) * 2  # Custom: scaled ReLU

% ============ MATLAB ============
% function A = apply_activation(Z, name)
%     switch name
%         case 'custom_scaled_relu'
%             A = max(0, Z) * 2;
%     end
% end

%% KEY DIFFERENCES: Python vs MATLAB

% 1. AUTOMATIC DIFFERENTIATION
%    Python: TensorFlow/PyTorch compute gradients automatically
%    MATLAB: Implement backpropagation manually (or use numerical gradients)

% 2. GPU ACCELERATION
%    Python: Automatic with TensorFlow/PyTorch on GPU
%    MATLAB: Requires Parallel Computing Toolbox for GPU (not implemented here)

% 3. DYNAMICS vs STATIC
%    Python: Dynamic graphs (PyTorch) or static (TensorFlow v1)
%    MATLAB: Static computation graph defined in structures

% 4. VECTORIZATION
%    Python: NumPy auto-vectorizes, TensorFlow optimizes
%    MATLAB: Must manually write vectorized operations

% 5. ECOSYSTEM
%    Python: Rich deep learning ecosystem (Hugging Face, etc.)
%    MATLAB: Smaller ML ecosystem, more scientific computing

%% EXTENDING THE SYSTEM

%% 1. Adding a new activation function

% Before: buildFNN.m uses only predefined activations
% After: Add to apply_activation() function

% Example: Add 'swish' activation (x * sigmoid(x))
function A = apply_activation_extended(Z, name)
    switch lower(name)
        case 'swish'
            A = Z .* (1 ./ (1 + exp(-Z)));  % x * sigmoid(x)
        case 'mish'
            % Mish: x * tanh(softplus(x))
            A = Z .* tanh(log(1 + exp(Z)));
        otherwise
            % Fall back to original
            % ... (existing cases)
    end
end

%% 2. Adding a new loss function

% Before: only MSE, MAE, Huber in computeLoss()
% After: Add new case

% Example: Add 'quantile' loss for quantile regression
function loss = computeLoss_extended(y_true, y_pred, loss_name, varargin)
    error = y_true - y_pred;
    
    switch lower(loss_name)
        case 'quantile'
            % Quantile loss: q * (y - y_pred) if y > y_pred
            %               (1-q) * (y_pred - y) otherwise
            q = 0.5;  % Median (0.5-quantile)
            if ~isempty(varargin)
                q = varargin{1};
            end
            loss = mean(max(q * error, (q-1) * error));
    end
end

%% 3. Adding a new optimizer

% Before: Adam, SGD, RMSprop, Adagrad in trainNetwork.m
% After: Add new case

% Example: Add 'AdamW' (Adam with decoupled weight decay)
function [net, opt] = adamwUpdate(net, dW, db, opt, weight_decay)
    opt.t = opt.t + 1;
    
    for layer = 1:length(dW)
        % Adam update (same as before)
        opt.m{layer} = opt.beta1 * opt.m{layer} + (1-opt.beta1) * dW{layer};
        opt.v{layer} = opt.beta2 * opt.v{layer} + (1-opt.beta2) * dW{layer}.^2;
        
        m_hat = opt.m{layer} / (1 - opt.beta1^opt.t);
        v_hat = opt.v{layer} / (1 - opt.beta2^opt.t);
        
        adam_update = m_hat ./ (sqrt(v_hat) + opt.epsilon);
        
        % Decoupled weight decay (key difference from Adam)
        net.weights{layer} = net.weights{layer} * (1 - weight_decay * opt.learning_rate);
        net.weights{layer} = net.weights{layer} - opt.learning_rate * adam_update;
        
        net.biases{layer} = net.biases{layer} - opt.learning_rate * db{layer};
    end
end

%% 4. Adding batch normalization

% Current: No batch normalization
% Extended: Add after activations in forward pass

% Add to predict.m forward pass:
function [y_pred, activations, z_vals, bn_stats] = predict_with_bn(net, X, varargin)
    % varargin: training (boolean), momentum (scalar)
    % Returns bn_stats for updating running statistics
    
    % For each layer after activation:
    % Z_normalized = (Z - batch_mean) / sqrt(batch_var + epsilon)
    % Z_scaled = gamma * Z_normalized + beta
    
    % During training: use batch statistics
    % During inference: use running statistics
end

%% 5. Adding L1/L2 regularization

% Before: No regularization
% After: Add weight decay to optimizer updates

function [net] = apply_regularization(net, lambda_l1, lambda_l2)
    for layer = 1:length(net.weights)
        % L1 regularization (additive penalty on gradients)
        % gradient += lambda_l1 * sign(weight)
        
        % L2 regularization (weight decay)
        % weight *= (1 - lambda_l2 * learning_rate)
        
        % Implementation in updateWeights:
        net.weights{layer} = net.weights{layer} * (1 - lambda_l2);
    end
end

%% 6. Adding learning rate scheduling

% Before: Fixed learning rate
% After: Vary learning rate during training

function adapted_lr = get_learning_rate(epoch, config)
    switch lower(config.lr_schedule)
        case 'constant'
            adapted_lr = config.learning_rate;
        case 'exponential_decay'
            decay_rate = 0.95;
            adapted_lr = config.learning_rate * (decay_rate ^ epoch);
        case 'cosine_annealing'
            % Cosine annealing: gradual decrease then reset
            T_max = config.epochs;
            adapted_lr = config.learning_rate * 0.5 * (1 + cos(pi * epoch / T_max));
        case 'step_decay'
            step_epochs = 10;
            drop = 0.5;
            steps = floor(epoch / step_epochs);
            adapted_lr = config.learning_rate * (drop ^ steps);
    end
end

%% 7. Adding early stopping

% Before: Always train for fixed epochs
% After: Stop if validation improves slowly

function should_stop = check_early_stopping(history, patience, min_delta)
    if length(history.val_loss) < patience + 1
        should_stop = false;
        return;
    end
    
    recent_losses = history.val_loss(end-patience:end);
    best_loss = min(recent_losses(1:end-1));
    current_loss = recent_losses(end);
    
    should_stop = (current_loss - best_loss) > -min_delta;
end

%% 8. Handling different data types and missing values

% Before: Simple NaN removal
% After: Advanced preprocessing

function [X, y, removed_info] = preprocess_data_advanced(X, y, method)
    removed_info = struct();
    
    % Option 1: Remove rows with ANY NaN
    valid = ~any(isnan(X), 2) & ~isnan(y);
    
    % Option 2: Interpolate missing values (for time series)
    % X = fillmissing(X, 'linear');
    
    % Option 3: Forward fill
    % X = fillmissing(X, 'previous');
    
    % Option 4: Replace with mean
    X_mean = mean(X, 1, 'omitnan');
    for i = 1:size(X, 2)
        nan_mask = isnan(X(:, i));
        X(nan_mask, i) = X_mean(i);
    end
    
    removed_info.rows_removed = sum(~valid);
    X = X(valid, :);
    y = y(valid);
end

%% PERFORMANCE OPTIMIZATION TIPS

% 1. VECTORIZATION (Most Important)
%    Slow:  for i = 1:size(X,1)
%             y(i) = X(i,:) * w + b;
%           end
%    Fast:  y = X * w + repmat(b, size(X,1), 1);

% 2. PREALLOCATION
%    Slow:  losses = [];
%           for i = 1:1000
%               losses = [losses, compute_loss()];
%           end
%    Fast:  losses = zeros(1000, 1);
%           for i = 1:1000
%               losses(i) = compute_loss();
%           end

% 3. AVOID READ/WRITE FROM DISK IN LOOPS
%    Batch process all data at once

% 4. USE BUILT-IN FUNCTIONS
%    Avoid reimplementing MATLAB built-ins

% 5. PROFILE CODE
%    profile on
%    ... code ...
%    profile off
%    profview

%% COMPARISON TABLE

fprintf('\n=== Python vs MATLAB Feature Comparison ===\n');
fprintf('Feature                    Python (TensorFlow)      MATLAB\n');
fprintf('%-27s %-25s %-20s\n', '----------', '----', '----');
fprintf('%-27s %-25s %-20s\n', 'Automatic differentiation', 'Yes (eager/graph)', 'No (manual)');
fprintf('%-27s %-25s %-20s\n', 'GPU support', 'Native ', 'Parallel Toolbox');
fprintf('%-27s %-25s %-20s\n', 'Batch normalization', 'Built-in', 'Custom');
fprintf('%-27s %-25s %-20s\n', 'Pre-trained models', 'Many', 'Few');
fprintf('%-27s %-25s %-20s\n', 'Deployment', 'Web/Mobile/Edge', 'Server/Desktop');
fprintf('%-27s %-25s %-20s\n', 'Learning curve', 'Moderate', 'Moderate');
fprintf('%-27s %-25s %-20s\n', 'Production readiness', 'High', 'Medium');

%% DEBUGGING TIPS

% 1. Check shapes at each layer
%    disp(size(Z))  % After linear transformation
%    disp(size(A))  % After activation

% 2. Verify gradients numerically
%    Compute analytical gradients
%    Compare with numerical gradients (central difference)

% 3. Monitor for NaN/Inf
%    assert(all(~isnan(Z(:))))
%    assert(all(~isinf(Z(:))))

% 4. Use visualizations
%    histogram(weights)     % Check weight distribution
%    plot(history)          % Check training curves

fprintf('\n=== End of Migration Guide ===\n');
