# Neural Networks

This project implements a simple feedforward neural network from scratch using NumPy, demonstrating the fundamental concepts of forward propagation, backpropagation, and gradient descent.

## Overview

The notebook contains a complete implementation of a neural network with:
- 1 input neuron
- 3 hidden layer neurons
- 1 output neuron
- Sigmoid activation functions
- Mean squared error loss function

## Network Architecture

The network structure consists of:
- **Input Layer**: Single input neuron (z₅)
- **Hidden Layer**: Three neurons (z₂, z₃, z₄) with sigmoid activation
- **Output Layer**: Single output neuron (z₁) with sigmoid activation

### Mathematical Formulation

**Forward Pass:**
```
z₅ = x (input)

Hidden Layer:
a₂ = w₅² · z₅
z₂ = sigmoid(a₂)

a₃ = w₅³ · z₅
z₃ = sigmoid(a₃)

a₄ = w₅⁴ · z₅
z₄ = sigmoid(a₄)

Output Layer:
a₁ = w₂¹ · z₂ + w₃¹ · z₃ + w₄¹ · z₄
z₁ = sigmoid(a₁)
```

**Activation Function:**
- Sigmoid: `f(x) = 1 / (1 + e^(-x))`
- Sigmoid Derivative: `f'(x) = f(x) · (1 - f(x))`

**Loss Function:**
- Mean Squared Error: `E = 0.5 · (target - z₁)²`

## Implementation Details

### Key Functions

1. **`sigmoid(x)`**: Implements the sigmoid activation function
2. **`back_sigmoid(x)`**: Computes the derivative of sigmoid for backpropagation
3. **`forward_pass(X, weights_input_hidden, weights_hidden_output)`**:
   - Computes network output through forward propagation
   - Returns output and hidden layer activations
4. **`backward_pass(...)`**:
   - Computes gradients using backpropagation
   - Returns weight gradients for both layers
5. **`train(X_train, y_train)`**:
   - Trains the network using gradient descent
   - Visualizes training loss and final predictions

### Backpropagation

The error gradients (δᵢ) are computed using the chain rule:

**Output Layer:**
```
δ₁ = -(target - z₁) · z₁ · (1 - z₁)
```

**Hidden Layer:**
```
δ₂ = δ₁ · w₂¹ · z₂ · (1 - z₂)
δ₃ = δ₁ · w₃¹ · z₃ · (1 - z₃)
δ₄ = δ₁ · w₄¹ · z₄ · (1 - z₄)
```

## Training Configuration

- **Learning Rate**: 0.1
- **Epochs**: 1000
- **Weight Initialization**: Random uniform distribution
- **Training Data**: 11 samples with x values from -3.0 to 3.0

### Training Data

| x    | y      |
|------|--------|
| -3.0 | 0.7312 |
| -2.0 | 0.7339 |
| -1.5 | 0.7438 |
| -1.0 | 0.7832 |
| -0.5 | 0.8903 |
| 0.0  | 0.9820 |
| 0.5  | 0.8114 |
| 1.0  | 0.5937 |
| 1.5  | 0.5219 |
| 2.0  | 0.5049 |
| 3.0  | 0.5002 |

## Visualizations

The notebook includes:
1. Training loss curve showing error reduction over epochs
2. Comparison plot of predictions vs. actual target values
3. Network architecture diagram (simple_nn.png)

## Usage

Run all cells in [script.ipynb](script.ipynb) to:
1. Define the network architecture and training functions
2. Train the network on the provided dataset
3. View training progress (printed every 100 epochs)
4. Generate loss and prediction visualizations
5. Display final learned weights

## Requirements

- NumPy
- Matplotlib
- IPython (for Image display)

## Key Learning Concepts

- Forward propagation through a multi-layer network
- Backpropagation algorithm for computing gradients
- Gradient descent optimization
- Implementation of neural networks from scratch without deep learning frameworks
