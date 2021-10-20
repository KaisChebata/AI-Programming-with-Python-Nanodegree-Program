"""
From before we saw that one weight update can be calculated as:

Δwi=ηδxi

with the error term δ as:

δ = (y−y^)f′(h) = (y−y^)f′(∑wixi)

Remember, in the above equation:
(y−y^) is the output error, 
and f′(h) refers to the derivative of the activation function, f(h). 
We'll call that derivative the output gradient.
"""

import numpy as np

# Defining the sigmoid function for activations
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Derivative of the sigmoid function
def sigmoid_prime(x):
    return (sigmoid(x) * (1 - sigmoid(x)))

# Input data
x = np.array([0.1, 0.3])

# Target
y = 0.2

# Input to output weights
weights = np.array([-0.8, 0.5])

# The learning rate, eta in the weight step equation
learnrate = 0.5

# the linear combination performed by the node (h in f(h) and f'(h))
h = x[0] * weights[0] + x[1] * weights[1]
# or h = np.dot(x, weights)

# The neural network output (y-hat)
nn_output = sigmoid(h)

#  output error (y - y-hat)
error = y - nn_output

# output gradient (f'(h))
out_grad = sigmoid_prime(h)

# error term (lowercase delta)
error_term = error * out_grad

# Gradient descent step 
del_w = [learnrate * error_term * x[0],
         learnrate * error_term * x[1]]
# or del_w = learnrate * error_term * x

print(f'data record are: x = {x} - target = {y}')
print(f'weights: {weights}')
print(f'h = x * weights = {h}')
print(f'f(h) = {nn_output}')
print(f'error output = (y - y^) = {error}')
print(f"output gradient = f'(h) = {out_grad}")
print(f'Gradient descent step (del_w): {del_w}')

