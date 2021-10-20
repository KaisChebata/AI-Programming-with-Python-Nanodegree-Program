import numpy as np

def sigmoid(x):
    """Calculate Sigmoid"""
    return (1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    """Calculate Sigmoid Prime"""
    return (sigmoid(x) * (1 - sigmoid(x)))

# features and targets
X = np.array([0.1, 0.3])
y = 1

# learing rate
learnrate = 0.5

# weights
weights_input_to_hidden = np.array([0.4, -0.2])
weights_hidden_to_output = np.array(0.1)
print(weights_hidden_to_output)

# hidden input
h = np.dot(X, weights_input_to_hidden).round(2)

# hidden output activation (a)
a = sigmoid(h)

print(f'features: {X}')
print(f'Weights Input to Hidden: {weights_input_to_hidden}')
print(f'hidden input => h = {h}')
print(f'activation of h => a = {a}')
print(f'Weights Hidden To Output: {weights_hidden_to_output}')

# output
output_y_hat = sigmoid(np.dot(a, weights_hidden_to_output))
print(f'Output y^ = {output_y_hat}')

# output error
error = y - output_y_hat

# error term for output
error_term_output = (error * sigmoid_prime(a * weights_hidden_to_output)).round(3)
print(f'error term for output = {error_term_output}')

# error term for hidden unit
error_term_hidden_unit = (weights_hidden_to_output * error_term_output * sigmoid_prime(a)).round(3)
print(f'Error Term Hidden Unit: {error_term_hidden_unit}')

# gradient descent steps. The hidden to output weight step
del_w_h_to_out = (learnrate * error_term_output * a).round(4)
print(f'del_w for hidden to output weight: {del_w_h_to_out}')

# gradient descent steps. The input to hidden weight step
del_w_input_to_h = (learnrate * error_term_hidden_unit * X)
print(f'del_w for input to hidden weight: {del_w_input_to_h}')


