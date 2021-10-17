"""
Question 1 of 2

Based on the lesson videos, 
let's define the combination of two new perceptrons as w1*0.4 + w2*0.6 + b. 
Which of the following values for the weights and the bias 
would result in the final probability of the point to be 0.88?

a. w1: 2, w2: 6, b: -2
b. w1: 3, w2: 5, b: -2.2
c. w1: 5, w2: 4, b: -3

Answer: b.
"""
import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def out_prob(features, weights, bias):
    return sigmoid(np.matmul(features, weights) + bias)

w1, w2, b = [2, 3, 5], [6, 5, 4], [-2, -2.2, -3]
features = np.array([0.4, 0.6])

for model in zip(w1, w2, b):
    weights, bias = np.array([model[0], model[1]]), model[2]
    
    print(
        f'final probability of the point of model '
        f'({model[0]}*0.4 + {model[1]}*0.6 {model[2]}): {out_prob(features, weights, bias)}'
        )
    
    # print(weights)
    # print(bias)

print(sigmoid(.7 + .8))
