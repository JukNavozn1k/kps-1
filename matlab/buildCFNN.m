function net = buildCFNN(input_dim, layer_units, activations, dropout_rates, varargin)
%BUILDCFNN Build a Cascade Feedforward Neural Network
%
% Cascade FNN connects input and previous layers to all subsequent layers
%
% Inputs:
%   input_dim: Number of input features
%   layer_units: Vector of units in each hidden layer
%   activations: Cell array of activation functions
%   dropout_rates: Vector of dropout rates for each layer
%   varargin: Optional name-value pairs
%     'include_input_to_hidden': Include input to all hidden layers (default: true)
%     'include_prev_to_hidden': Include previous layers to next (default: true)
%     'include_input_to_output': Include input to output (default: true)
%     'include_hidden_to_output': Include hidden layers to output (default: true)
%
% Returns:
%   net: Neural network model (structure)

    % Parse options
    p = inputParser;
    addParameter(p, 'include_input_to_hidden', true, @islogical);
    addParameter(p, 'include_prev_to_hidden', true, @islogical);
    addParameter(p, 'include_input_to_output', true, @islogical);
    addParameter(p, 'include_hidden_to_output', true, @islogical);
    parse(p, varargin{:});
    
    opts = p.Results;
    
    net = struct();
    net.type = 'CFNN';
    net.layers = {};
    net.weights = {};
    net.biases = {};
    net.activations = activations;
    net.input_dim = input_dim;
    net.output_dim = 1;
    net.connectivity = opts;
    
    % Track layer dimensions for cascade connections
    layer_dims = [input_dim]; % Start with input dimension
    
    % Initialize hidden layers
    for i = 1:length(layer_units)
        % Calculate input dimension to this layer
        % In cascade, current layer receives:
        % - Input (if include_input_to_hidden)
        % - Previous layers (if include_prev_to_hidden)
        
        input_to_this_layer = 0;
        
        if opts.include_input_to_hidden
            input_to_this_layer = input_to_this_layer + input_dim;
        end
        
        if opts.include_prev_to_hidden && i > 1
            % Sum all previous layer outputs
            input_to_this_layer = input_to_this_layer + sum(layer_units(1:i-1));
        end
        
        if input_to_this_layer == 0
            input_to_this_layer = input_dim;
        end
        
        layer = struct();
        layer.type = 'Dense_Cascade';
        layer.units = layer_units(i);
        layer.activation = activations{i};
        layer.dropout = dropout_rates(i);
        layer.input_dim = input_to_this_layer;
        
        % Initialize weights
        if strcmp(activations{i}, 'relu')
            scale = sqrt(2 / input_to_this_layer);
        else
            scale = sqrt(1 / input_to_this_layer);
        end
        
        W = randn(input_to_this_layer, layer_units(i)) * scale;
        b = zeros(1, layer_units(i));
        
        net.weights{end+1} = W;
        net.biases{end+1} = b;
        net.layers{end+1} = layer;
        
        layer_dims = [layer_dims, layer_units(i)];
    end
    
    % Output layer
    output_input_dim = 0;
    if opts.include_input_to_output
        output_input_dim = output_input_dim + input_dim;
    end
    if opts.include_hidden_to_output
        output_input_dim = output_input_dim + sum(layer_units);
    end
    
    if output_input_dim == 0
        output_input_dim = input_dim;
    end
    
    W_out = randn(output_input_dim, net.output_dim) * sqrt(1 / output_input_dim);
    b_out = zeros(1, net.output_dim);
    
    net.weights{end+1} = W_out;
    net.biases{end+1} = b_out;
    
    output_layer = struct();
    output_layer.type = 'Dense_Cascade';
    output_layer.units = net.output_dim;
    output_layer.activation = 'linear';
    output_layer.dropout = 0;
    output_layer.input_dim = output_input_dim;
    net.layers{end+1} = output_layer;
    
    net.n_layers = length(net.layers);
    net.layer_dims = layer_dims;
    
end
