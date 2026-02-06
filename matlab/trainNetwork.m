function [net, history] = trainNetwork(net, X_train, y_train, X_val, y_val, config)
%TRAINNETWORK Train neural network with gradient descent optimization
%
% Inputs:
%   net: Neural network structure
%   X_train, y_train: Training data
%   X_val, y_val: Validation data
%   config: Configuration structure
%
% Returns:
%   net: Trained network
%   history: Training history

    n_train = length(y_train);
    n_epochs = config.epochs;
    batch_size = config.batch_size;
    learning_rate = config.learning_rate;
    optimizer_name = config.optimizer;
    loss_name = config.loss_fn;
    
    % Initialize optimizer states
    optimizer = struct();
    optimizer.name = optimizer_name;
    optimizer.learning_rate = learning_rate;
    
    switch lower(optimizer_name)
        case 'adam'
            optimizer.beta1 = 0.9;
            optimizer.beta2 = 0.999;
            optimizer.epsilon = 1e-8;
            optimizer.m = cellfun(@zeros, cellfun(@size, net.weights, 'UniformOutput', false), ...
                                  cellfun(@(x) 0, net.weights, 'UniformOutput', false), 'UniformOutput', false);
            optimizer.v = cellfun(@zeros, cellfun(@size, net.weights, 'UniformOutput', false), ...
                                  cellfun(@(x) 0, net.weights, 'UniformOutput', false), 'UniformOutput', false);
            optimizer.t = 0;  % Time step
            
        case 'sgd'
            % SGD with optional momentum
            optimizer.momentum = 0.0;
            optimizer.velocity = cellfun(@zeros, cellfun(@size, net.weights, 'UniformOutput', false), ...
                                         cellfun(@(x) 0, net.weights, 'UniformOutput', false), 'UniformOutput', false);
            
        case 'rmsprop'
            optimizer.decay = 0.99;
            optimizer.epsilon = 1e-8;
            optimizer.ms = cellfun(@zeros, cellfun(@size, net.weights, 'UniformOutput', false), ...
                                   cellfun(@(x) 0, net.weights, 'UniformOutput', false), 'UniformOutput', false);
            
        case 'adagrad'
            optimizer.epsilon = 1e-8;
            optimizer.accumulator = cellfun(@zeros, cellfun(@size, net.weights, 'UniformOutput', false), ...
                                            cellfun(@(x) 0, net.weights, 'UniformOutput', false), 'UniformOutput', false);
    end
    
    % Training history
    history = struct();
    history.train_loss = [];
    history.val_loss = [];
    history.train_mae = [];
    history.val_mae = [];
    
    % Main training loop
    fprintf('Starting training for %d epochs...\n', n_epochs);
    
    for epoch = 1:n_epochs
        % Shuffle training data
        idx = randperm(n_train);
        X_train_shuffled = X_train(idx, :);
        y_train_shuffled = y_train(idx);
        
        % Batch processing
        n_batches = ceil(n_train / batch_size);
        batch_losses = [];
        
        for batch = 1:n_batches
            start_idx = (batch - 1) * batch_size + 1;
            end_idx = min(batch * batch_size, n_train);
            
            X_batch = X_train_shuffled(start_idx:end_idx, :);
            y_batch = y_train_shuffled(start_idx:end_idx);
            
            % Forward pass
            y_pred = predict(net, X_batch);
            
            % Compute loss
            batch_loss = computeLoss(y_batch, y_pred, loss_name);
            batch_losses = [batch_losses, batch_loss];
            
            % Backward pass (compute gradients)
            [dW, db] = computeGradients(net, X_batch, y_batch, y_pred, loss_name);
            
            % Update weights
            [net, optimizer] = updateWeights(net, dW, db, optimizer);
        end
        
        % Validation
        y_train_pred = predict(net, X_train);
        y_val_pred = predict(net, X_val);
        
        train_loss = computeLoss(y_train, y_train_pred, loss_name);
        val_loss = computeLoss(y_val, y_val_pred, loss_name);
        
        train_mae = mean(abs(y_train - y_train_pred));
        val_mae = mean(abs(y_val - y_val_pred));
        
        history.train_loss = [history.train_loss, train_loss];
        history.val_loss = [history.val_loss, val_loss];
        history.train_mae = [history.train_mae, train_mae];
        history.val_mae = [history.val_mae, val_mae];
        
        if mod(epoch, 10) == 0 || epoch == 1
            fprintf('Epoch %3d/%d - Loss: %.6f | Val Loss: %.6f | MAE: %.4f | Val MAE: %.4f\n', ...
                epoch, n_epochs, train_loss, val_loss, train_mae, val_mae);
        end
    end
    
    fprintf('Training completed!\n');
    
end

function loss = computeLoss(y_true, y_pred, loss_name)
%COMPUTELOSS Compute loss value
    
    error = y_true - y_pred;
    
    switch lower(loss_name)
        case 'mse'
            loss = mean(error.^2);
        case 'mae'
            loss = mean(abs(error));
        case 'huber'
            % Huber loss with delta = 1.0
            delta = 1.0;
            abs_error = abs(error);
            loss = mean((abs_error <= delta) .* (0.5 * error.^2) + ...
                        (abs_error > delta) .* (delta * (abs_error - 0.5 * delta)));
        otherwise
            error('Unknown loss function: %s', loss_name);
    end
    
end

function [dW, db] = computeGradients(net, X_batch, y_batch, y_pred, loss_name)
%COMPUTEGRADIENTS Compute gradients using backpropagation
    
    n_samples = size(X_batch, 1);
    n_layers = length(net.weights);
    
    % Initialize gradient arrays
    dW = cell(n_layers, 1);
    db = cell(n_layers, 1);
    
    % Compute loss derivative
    error = y_pred - y_batch;
    
    switch lower(loss_name)
        case 'mse'
            dL_dy = 2 * error / n_samples;
        case 'mae'
            dL_dy = sign(error) / n_samples;
        case 'huber'
            delta = 1.0;
            dL_dy = zeros(size(error));
            mask_small = abs(error) <= delta;
            mask_large = abs(error) > delta;
            dL_dy(mask_small) = error(mask_small);
            dL_dy(mask_large) = delta * sign(error(mask_large));
            dL_dy = dL_dy / n_samples;
    end
    
    % Backward pass (simplified numerical gradients for stability)
    % For production, implement full backpropagation with chain rule
    epsilon = 1e-5;
    
    for layer = 1:n_layers
        dW{layer} = zeros(size(net.weights{layer}));
        db{layer} = zeros(size(net.biases{layer}));
        
        % Numerical gradients
        for i = 1:size(net.weights{layer}, 1)
            for j = 1:size(net.weights{layer}, 2)
                net_plus = net;
                net_plus.weights{layer}(i,j) = net.weights{layer}(i,j) + epsilon;
                y_plus = predict(net_plus, X_batch);
                loss_plus = computeLoss(y_batch, y_plus, loss_name);
                
                net_minus = net;
                net_minus.weights{layer}(i,j) = net.weights{layer}(i,j) - epsilon;
                y_minus = predict(net_minus, X_batch);
                loss_minus = computeLoss(y_batch, y_minus, loss_name);
                
                dW{layer}(i,j) = (loss_plus - loss_minus) / (2 * epsilon);
            end
        end
        
        for j = 1:length(net.biases{layer})
            net_plus = net;
            net_plus.biases{layer}(j) = net.biases{layer}(j) + epsilon;
            y_plus = predict(net_plus, X_batch);
            loss_plus = computeLoss(y_batch, y_plus, loss_name);
            
            net_minus = net;
            net_minus.biases{layer}(j) = net.biases{layer}(j) - epsilon;
            y_minus = predict(net_minus, X_batch);
            loss_minus = computeLoss(y_batch, y_minus, loss_name);
            
            db{layer}(j) = (loss_plus - loss_minus) / (2 * epsilon);
        end
    end
    
end

function [net, optimizer] = updateWeights(net, dW, db, optimizer)
%UPDATEWEIGHTS Update network weights using optimizer
    
    switch lower(optimizer.name)
        case 'adam'
            [net, optimizer] = adamUpdate(net, dW, db, optimizer);
        case 'sgd'
            [net, optimizer] = sgdUpdate(net, dW, db, optimizer);
        case 'rmsprop'
            [net, optimizer] = rmspropUpdate(net, dW, db, optimizer);
        case 'adagrad'
            [net, optimizer] = adagradUpdate(net, dW, db, optimizer);
    end
    
end

function [net, opt] = adamUpdate(net, dW, db, opt)
%ADAMUPDATE Adam optimizer update
    
    opt.t = opt.t + 1;
    
    for layer = 1:length(dW)
        % Weight update
        opt.m{layer} = opt.beta1 * opt.m{layer} + (1 - opt.beta1) * dW{layer};
        opt.v{layer} = opt.beta2 * opt.v{layer} + (1 - opt.beta2) * dW{layer}.^2;
        
        m_hat = opt.m{layer} / (1 - opt.beta1^opt.t);
        v_hat = opt.v{layer} / (1 - opt.beta2^opt.t);
        
        net.weights{layer} = net.weights{layer} - opt.learning_rate * m_hat ./ (sqrt(v_hat) + opt.epsilon);
        
        % Bias update (simpler)
        net.biases{layer} = net.biases{layer} - opt.learning_rate * db{layer};
    end
    
end

function [net, opt] = sgdUpdate(net, dW, db, opt)
%SGDUPDATE SGD with momentum
    
    for layer = 1:length(dW)
        if opt.momentum > 0
            opt.velocity{layer} = opt.momentum * opt.velocity{layer} - opt.learning_rate * dW{layer};
            net.weights{layer} = net.weights{layer} + opt.velocity{layer};
        else
            net.weights{layer} = net.weights{layer} - opt.learning_rate * dW{layer};
        end
        
        net.biases{layer} = net.biases{layer} - opt.learning_rate * db{layer};
    end
    
end

function [net, opt] = rmspropUpdate(net, dW, db, opt)
%RMSPROPUPDATE RMSprop optimizer update
    
    for layer = 1:length(dW)
        opt.ms{layer} = opt.decay * opt.ms{layer} + (1 - opt.decay) * dW{layer}.^2;
        net.weights{layer} = net.weights{layer} - opt.learning_rate * dW{layer} ./ (sqrt(opt.ms{layer}) + opt.epsilon);
        net.biases{layer} = net.biases{layer} - opt.learning_rate * db{layer};
    end
    
end

function [net, opt] = adagradUpdate(net, dW, db, opt)
%ADAGRADUPDATE Adagrad optimizer update
    
    for layer = 1:length(dW)
        opt.accumulator{layer} = opt.accumulator{layer} + dW{layer}.^2;
        net.weights{layer} = net.weights{layer} - opt.learning_rate * dW{layer} ./ (sqrt(opt.accumulator{layer}) + opt.epsilon);
        net.biases{layer} = net.biases{layer} - opt.learning_rate * db{layer};
    end
    
end
