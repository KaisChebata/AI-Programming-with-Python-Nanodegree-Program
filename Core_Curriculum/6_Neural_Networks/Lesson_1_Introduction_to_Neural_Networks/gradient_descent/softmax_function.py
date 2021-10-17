"""
Sofmax function:
Let's say that we have a linear model that give us the following scores:
Z1,...., Zn each score for each of the classes

P(classi) = e^Zi / e^Z1 + .... + e^Zn
"""

import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.

# Trying for L=[5,6,7].
# The correct answer is
# [0.09003057317038046, 0.24472847105479764, 0.6652409557748219]

def softmax(L):
    # denominator = sum(map(lambda x: np.exp(x), L))
    denominator = sum(np.exp(L))
    return [np.exp(i) / denominator for i in L]

def sigmoid(val: float) -> float:
    return (1 / (1 + np.exp(-val)))

L=[5,6,7]
print(f'get softmax for {L}')
print(softmax(L))

l2 = [5, 6]
print(f'get softmax for {l2}')
print(softmax(l2))
print(f'get sigmoid for {l2}')
print([sigmoid(i) for i in l2])