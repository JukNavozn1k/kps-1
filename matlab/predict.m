function [y_pred, activations, z_vals] = predict(net, X)
%PREDICT Forward pass through neural network
%
% Inputs:
%   net: Neural network structure
%   X: Input data (n_samples x n_features)
%
% Returns:
%   y_pred: Predictions (n_samples x output_dim)
%   activations: Cell array of activation values at each layer (optional)
%   z_vals: Cell array of pre-activation values (optional)

    if strcmp(net.type, 'FNN')
        [y_pred, activations, z_vals] = predict_fnn(net, X);
    elseif strcmp(net.type, 'CFNN')
        [y_pred, activations, z_vals] = predict_cfnn(net, X);
    else
        error('Unknown network type: %s', net.type);
    end
    
end

function [y_pred, activations, z_vals] = predict_fnn(net, X)
%PREDICT_FNN Forward pass for standard FNN
    
    n_samples = size(X, 1);
    n_layers = length(net.weights);
    
    activations = cell(n_layers, 1);
    z_vals = cell(n_layers, 1);
    
    % Forward pass through layers
    A = X;
    activations{1} = A;
    
    for i = 1:n_layers-1
        Z = A * net.weights{i} + repmat(net.biases{i}, n_samples, 1);
        z_vals{i} = Z;
        
        % Apply activation
        A = apply_activation(Z, net.layers{i}.activation);
        
        % Apply dropout (only during prediction with rate info)
        if net.layers{i}.dropout > 0 && nargout > 1
            % During forward pass without training flag, keep all units
        end
        
        activations{i+1} = A;
    end
    
    % Output layer (linear activation)
    Z = A * net.weights{end} + repmat(net.biases{end}, n_samples, 1);
    z_vals{end} = Z;
    y_pred = Z;
    activations{end} = y_pred;
    
end

function [y_pred, activations, z_vals] = predict_cfnn(net, X)
%PREDICT_CFNN Forward pass for Cascade FNN
    
    n_samples = size(X, 1);
    n_hidden = length(net.layers) - 1;  % Exclude output layer
    
    activations = {};
    z_vals = {};
    hidden_outputs = {};
    
    % Forward pass through hidden layers
    for i = 1:n_hidden
        layer = net.layers{i};
        
        % Concatenate inputs for cascade
        to_concat = {};
        
        if net.connectivity.include_input_to_hidden
            to_concat{end+1} = X;
        end
        
        if net.connectivity.include_prev_to_hidden && ~isempty(hidden_outputs)
            to_concat = [to_concat, hidden_outputs];
        end
        
        if isempty(to_concat)
            to_concat = {X};
        end
        
        % Concatenate along feature dimension
        if length(to_concat) == 1
            A_input = to_concat{1};
        else
            A_input = horzcat(to_concat{:});
        end
        
        % Linear transformation
        Z = A_input * net.weights{i} + repmat(net.biases{i}, n_samples, 1);
        z_vals{end+1} = Z;
        
        % Apply activation
        A = apply_activation(Z, layer.activation);
        activations{end+1} = A;
        hidden_outputs{end+1} = A;
    end
    
    % Output layer with cascade connections
    to_concat = {};
    if net.connectivity.include_input_to_output
        to_concat{end+1} = X;
    end
    if net.connectivity.include_hidden_to_output
        to_concat = [to_concat, hidden_outputs];
    end
    
    if isempty(to_concat)
        to_concat = {X};
    end
    
    if length(to_concat) == 1
        A_input = to_concat{1};
    else
        A_input = horzcat(to_concat{:});
    end
    
    Z = A_input * net.weights{end} + repmat(net.biases{end}, n_samples, 1);
    z_vals{end+1} = Z;
    y_pred = Z;
    
end

function A = apply_activation(Z, activation_name)
%APPLY_ACTIVATION Apply activation function
    
    switch lower(activation_name)
        case 'relu'
            A = max(0, Z);
        case 'tanh'
            A = tanh(Z);
        case 'sigmoid'
            A = 1 ./ (1 + exp(-Z));
        case 'elu'
            A = Z;
            A(Z < 0) = exp(Z(Z < 0)) - 1;
        case 'selu'
            % SELU: Scaled ELU
            lambda = 1.0507;
            alpha = 1.6733;
            A = Z;
            A(Z < 0) = lambda * alpha * (exp(Z(Z < 0)) - 1);
            A(Z >= 0) = lambda * Z(Z >= 0);
        case 'gelu'
            % Approximate GELU
            A = Z .* (0.5 * (1 + tanh(sqrt(2/pi) * (Z + 0.044715 * Z.^3))));
        case 'linear'
            A = Z;
        otherwise
            error('Unknown activation function: %s', activation_name);
    end
    
end
