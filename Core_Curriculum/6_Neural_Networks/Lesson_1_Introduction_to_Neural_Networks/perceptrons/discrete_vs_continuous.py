# from math import exp, log10
import numpy as np

"""
Quiz Question

The sigmoid function is defined as sigmoid(x) = 1/(1+e^-x). 
If the score is defined by (4x1 + 5x2 - 9 = score), 
then which of the following points has exactly a 50% probability 
of being blue or red? (Choose all that are correct.)

sigmoid(x) = 1 / (1 + exp(-x))

points:
(1, 1)
(2, 4)
(5, -5)
(-4, 5)
"""
def sigmoid(X, W, b):
    linear_value = (np.matmul(X, W) + b)[0]
    return (1 / (1 + np.exp(-linear_value)))


points = [(1, 1), (2, 4), (5, -5), (-4, 5)]
W = np.array([4, 5]).reshape(2, 1)
b = -9

for point in points:
    point_prop = sigmoid(point, W, b)
    print(f'{point} has probability with: {point_prop}')

print(np.exp(-500))
# sigmoid(x) = 1 / 1 + exp(-x)
