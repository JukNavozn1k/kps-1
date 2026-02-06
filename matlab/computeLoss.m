function loss = computeLoss(y_true, y_pred, loss_name)
%COMPUTELOSS Compute loss value between predictions and targets
%
% Inputs:
%   y_true: True labels (n x 1)
%   y_pred: Predictions (n x 1)
%   loss_name: Loss function name ('mse', 'mae', 'huber')
%
% Returns:
%   loss: Scalar loss value

    if nargin < 3
        loss_name = 'mse';
    end
    
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
            huber_values = (abs_error <= delta) .* (0.5 * error.^2) + ...
                           (abs_error > delta) .* (delta * (abs_error - 0.5 * delta));
            loss = mean(huber_values);
        otherwise
            error('Unknown loss function: %s', loss_name);
    end
    
end
