# Football Match Prediction System - MATLAB Version

This is a MATLAB implementation of a neural network-based football match prediction system, equivalent to the Python version using TensorFlow/Keras and Streamlit.

## Overview

This system predicts football match outcomes (goals scored) using fully connected neural networks with two architectures:

- **FNN (Feedforward Neural Network)**: Standard multi-layer perceptron
- **CFNN (Cascade Feedforward Neural Network)**: Extends FNN with cascade connections from input and previous layers to all subsequent layers

## Project Structure

```
football_prediction_main.m      % Main script - entry point
load_and_prepare_data.m         % Data loading and preprocessing
buildFNN.m                      % FNN architecture builder
buildCFNN.m                     % CFNN architecture builder
predict.m                       % Forward pass (prediction)
trainNetwork.m                  % Training loop with optimizers
computeLoss.m                   % Loss function implementations
```

## Features

### Supported Models
- **FNN**: Traditional feedforward architecture
- **CFNN**: Cascade connections for enhanced feature flow

### Supported Activations
- ReLU
- Tanh
- Sigmoid
- ELU (Exponential Linear Unit)
- SELU (Scaled ELU)
- GELU (Gaussian Error Linear Unit)
- Linear

### Supported Optimizers
- **Adam**: Adaptive Moment Estimation (recommended)
- **SGD**: Stochastic Gradient Descent with optional momentum
- **RMSprop**: Root Mean Square Propagation
- **Adagrad**: Adaptive Gradient

### Loss Functions
- **MSE**: Mean Squared Error (for regression)
- **MAE**: Mean Absolute Error (robust to outliers)
- **Huber**: Huber loss (balanced MSE/MAE)

### Data Preprocessing
- Automatic feature standardization (zero-mean, unit variance)
- Optional target standardization
- Missing value handling
- Train/validation split

## Usage

### Quick Start

```matlab
% Open MATLAB
% Navigate to the project directory
cd d:\kps-1

% Run the main script
football_prediction_main
```

### Configuration

Edit the `football_prediction_main.m` script to modify:

```matlab
config.data_file = 'ftball.csv';           % CSV file path
config.target = 'total_goals';             % Target variable
config.val_fraction = 0.2;                 % Validation set fraction
config.standardize_X = true;               % Normalize features
config.standardize_y = false;              % Normalize target

config.model_type = 'FNN';                 % 'FNN' or 'CFNN'
config.layers = [32, 16];                  % Hidden layer sizes
config.activations = {'relu', 'relu'};     % Activation functions
config.dropout = [0.0, 0.0];               % Dropout rates

config.optimizer = 'Adam';                 % Optimization algorithm
config.learning_rate = 5e-4;               % Learning rate
config.loss_fn = 'mse';                    % Loss function
config.epochs = 100;                       % Training epochs
config.batch_size = 32;                    % Batch size
config.seed = 42;                          % Random seed
```

### Example: Train Different Configurations

```matlab
% Example 1: FNN with Adam optimizer
config.model_type = 'FNN';
config.optimizer = 'Adam';
[X_train, y_train, X_val, y_val, feature_names, ~, ~, ~] = load_and_prepare_data(config);
net = buildFNN(size(X_train, 2), [64, 32, 16], {'relu', 'relu', 'relu'}, [0, 0, 0]);
[net, history] = trainNetwork(net, X_train, y_train, X_val, y_val, config);

% Example 2: CFNN with SGD optimizer
config.model_type = 'CFNN';
config.optimizer = 'SGD';
config.learning_rate = 1e-3;
net_cascade = buildCFNN(size(X_train, 2), [32, 16], {'relu', 'relu'}, [0, 0]);
[net_cascade, history_cascade] = trainNetwork(net_cascade, X_train, y_train, X_val, y_val, config);
```

## File Descriptions

### football_prediction_main.m
Main entry point that orchestrates the entire pipeline:
1. Loads configuration
2. Loads and preprocesses data
3. Builds network architecture
4. Trains the model
5. Evaluates performance
6. Generates visualization plots

### load_and_prepare_data.m
Handles data I/O and preprocessing:
- Reads CSV file using `readtable`
- Extracts numeric features
- Handles missing values
- Performs feature/target standardization
- Creates train/validation split

### buildFNN.m
Constructs a standard feedforward neural network:
- Initializes weights using He/Xavier initialization
- Configures activation functions per layer
- Sets up dropout if specified
- Creates output layer (linear activation)

### buildCFNN.m
Constructs a cascade feedforward neural network:
- Extends standard FNN with cascade connections
- Input can connect to all hidden layers
- Previous layers can feed into next layer
- Configurable connectivity options:
  - `include_input_to_hidden`: Input → all hidden layers
  - `include_prev_to_hidden`: Previous layers → next layer
  - `include_input_to_output`: Input → output
  - `include_hidden_to_output`: Hidden layers → output

### predict.m
Forward pass implementation:
- Routes to FNN or CFNN prediction based on network type
- Returns predictions, activations, and pre-activation values
- Implements all supported activation functions

### trainNetwork.m
Complete training pipeline:
- Initializes optimizer state
- Implements training loop with batching
- Computes gradients via backpropagation
- Updates weights using selected optimizer
- Tracks training history
- Performs validation at each epoch

**Includes optimizer implementations:**
- `adamUpdate`: Adam optimizer
- `sgdUpdate`: SGD with momentum
- `rmspropUpdate`: RMSprop
- `adagradUpdate`: Adagrad

### computeLoss.m
Loss function implementations:
- MSE: Sum of squared errors
- MAE: Sum of absolute errors
- Huber: Robust combination of MSE/MAE

## Performance Metrics

The system reports:
- **Training/Validation Loss**: Primary optimization metric
- **MAE**: Mean Absolute Error (interpretable units)
- **RMSE**: Root Mean Squared Error

Example output:
```
Training set size: 300 x 15
Validation set size: 100 x 15
Features: goal_diff, possession, shots, shots_on_target, ...

Model built successfully!

Training model...
Epoch  10/100 - Loss: 0.825443 | Val Loss: 0.912154 | MAE: 0.6234 | Val MAE: 0.6751
Epoch  20/100 - Loss: 0.543221 | Val Loss: 0.678234 | MAE: 0.4923 | Val MAE: 0.5612
...

Training Loss: 0.341234 | MAE: 0.3452 | RMSE: 0.5842
Validation Loss: 0.456789 | MAE: 0.4123 | RMSE: 0.6754
```

## Visualization

The script generates plots:
1. **Training Curves**: Loss vs epoch (train/validation)
2. **Prediction Scatter**: Actual vs predicted values
3. **Residuals**: Distribution of prediction errors

## Differences from Python Version

| Aspect | Python | MATLAB |
|--------|--------|--------|
| Framework | TensorFlow/Keras | Pure MATLAB |
| Interface | Streamlit web app | Scripts/functions |
| Distribution | Automatic | Numerical gradients (simplified) |
| Deployment | Web server | Desktop/batch |
| Parallelization | Built-in GPU | CPU-based |

## Requirements

- MATLAB R2018b or later (for `readtable`, `inputParser`)
- Statistics and Machine Learning Toolbox (optional, for enhanced stats)
- Data file: `ftball.csv` in working directory

## Data Format

Expected CSV format:
```
feature1, feature2, ... , target_variable
0.5, 0.2, ..., 2.0
1.2, 0.8, ..., 1.5
...
```

All columns should be numeric. Categorical variables should be encoded to numeric values before loading.

## Tips for Best Results

1. **Feature Normalization**: Keep `standardize_X = true` for stable training
2. **Learning Rate**: Start with 1e-3 to 1e-4 and adjust based on loss curves
3. **Batch Size**: Larger batches (128-256) for stable gradients, smaller (32-64) for exploration
4. **Epochs**: Monitor validation loss for early stopping
5. **Architecture**: Start simple (2-3 layers, 32-64 units) and increase if underfitting
6. **Optimizer**: Adam is generally robust; use SGD with lower learning rates for fine-tuning

## Extending the System

To add new features:

1. **New Loss Function**: Edit `computeLoss.m`
2. **New Optimizer**: Add update function in `trainNetwork.m`, add case to `updateWeights`
3. **New Activation**: Add case to `apply_activation` in `predict.m`
4. **Custom Data Loading**: Modify `load_and_prepare_data.m`
5. **Batch Normalization**: Add in `predict.m` forward pass
6. **Regularization**: Implement L1/L2 in weight updates

## Troubleshooting

**Issue**: Training loss not decreasing
- Solution: Reduce learning rate (try 1e-5), check data normalization

**Issue**: NaN/Inf in predictions
- Solution: Check for extreme values in data, reduce learning rate, verify activation functions

**Issue**: Out of memory
- Solution: Reduce batch size, reduce layer sizes, use fewer samples

**Issue**: Slow training
- Solution: Increase batch size, use fewer layers, reduce epochs for testing

## Future Improvements

- [ ] Full analytical backpropagation (replace numerical gradients)
- [ ] Batch normalization
- [ ] L1/L2 regularization
- [ ] Learning rate scheduling
- [ ] Early stopping
- [ ] Model checkpointing
- [ ] GPU acceleration with Parallel Computing Toolbox
- [ ] Convolutional layers for image data
- [ ] Recurrent layers for sequential data
- [ ] GUI dashboard similar to Streamlit

## License

Same as original Python implementation (see LICENSE file)

## References

- Deep Learning Fundamentals: https://www.deeplearningbook.org/
- Neural Network Optimization: Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
- Cascade Networks: Fahlman & Lebiere (1990) "The Cascade Correlation Learning Architecture"
