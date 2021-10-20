import numpy as np

def sigmoid(x):
    """Calculate Sigmoid"""
    return (1 / (1 + np.exp(-x)))

# Network size
N_INPUT, N_HIDDEN, N_OUTPUT = 4, 3, 2

# seed the randomness
np.random.seed(42)

# make fake data
X = np.random.randn(4)
print(f'Dataset: {X}')
"""
conceptual arch
x1

x2      h1
        h2      o1
x3      h3      o2

x4      
"""
weights_input_to_hidden = np.random.normal(loc=0, scale=0.1, size=(N_INPUT, N_HIDDEN))
weights_hidden_to_output = np.random.normal(loc=0, scale=0.1, size=(N_HIDDEN, N_OUTPUT))

# TODO: Make a forward pass through the network

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)
print(f'hidden_layer_in: {hidden_layer_in}')
print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)
print(f'output_layer_in: {output_layer_in}')
print('Output-layer Output:')
print(output_layer_out)