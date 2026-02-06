function net = buildFNN(input_dim, layer_units, activations, dropout_rates)
%BUILDFNN Build a Feedforward Neural Network
%
% Inputs:
%   input_dim: Number of input features
%   layer_units: Vector of units in each hidden layer
%   activations: Cell array of activation functions ('relu', 'tanh', 'sigmoid', 'linear')
%   dropout_rates: Vector of dropout rates for each layer
%
% Returns:
%   net: Neural network model (structure)

    net = struct();
    net.type = 'FNN';
    net.layers = {};
    net.weights = {};
    net.biases = {};
    net.activations = activations;
    net.input_dim = input_dim;
    net.output_dim = 1;
    
    % Initialize layers
    current_dim = input_dim;
    
    for i = 1:length(layer_units)
        layer = struct();
        layer.type = 'Dense';
        layer.units = layer_units(i);
        layer.activation = activations{i};
        layer.dropout = dropout_rates(i);
        
        % Initialize weights using He initialization for ReLU
        if strcmp(activations{i}, 'relu')
            scale = sqrt(2 / current_dim);
        else
            scale = sqrt(1 / current_dim);
        end
        
        W = randn(current_dim, layer_units(i)) * scale;
        b = zeros(1, layer_units(i));
        
        net.weights{end+1} = W;
        net.biases{end+1} = b;
        net.layers{end+1} = layer;
        
        current_dim = layer_units(i);
    end
    
    % Output layer
    W_out = randn(current_dim, net.output_dim) * sqrt(1 / current_dim);
    b_out = zeros(1, net.output_dim);
    
    net.weights{end+1} = W_out;
    net.biases{end+1} = b_out;
    
    output_layer = struct();
    output_layer.type = 'Dense';
    output_layer.units = net.output_dim;
    output_layer.activation = 'linear';
    output_layer.dropout = 0;
    net.layers{end+1} = output_layer;
    
    net.n_layers = length(net.layers);
    
end

function net_str = neuralNetworkToString(net)
%NEURALNETWORKTOSTRING Return string representation of network architecture
    net_str = sprintf('FNN Network:\n');
    net_str = [net_str, sprintf('  Input: %d\n', net.input_dim)];
    
    for i = 1:length(net.layers)
        layer = net.layers{i};
        net_str = [net_str, sprintf('  Layer %d: %d units, %s\n', ...
            i, layer.units, layer.activation)];
        if layer.dropout > 0
            net_str = [net_str, sprintf('    Dropout: %.2f\n', layer.dropout)];
        end
    end
    
    net_str = [net_str, sprintf('  Output: 1\n')];
end
