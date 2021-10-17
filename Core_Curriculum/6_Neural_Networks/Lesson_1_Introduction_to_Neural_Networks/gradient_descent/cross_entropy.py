"""
Quiz: Coding Cross-entropy

Now, time to shine! 
Let's code the formula for cross-entropy in Python. 
As in the video, Y in the quiz is for the category, and P is the probability.

Trying for Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5].
The correct answer is
4.8283137373
"""

"""
let's assume that Y and P as follows:
Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5]
let's take pairs of Y and P (i.e. using zip function)
[(1, 0.4), (0, 0.6), (1, 0.1), (1, 0.5)]
cross_entropy = - sum(yiln(pi) + (1-yi)ln(1-pi))
we can iterate through zip(Y, P) and for each pair we can calculate formula
"""
import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    # cross_entropy_lst = []
    # for y, p in zip(Y, P):
    #     if y == 1:
    #         cross_entropy_lst.append(np.log(p))
    #     else:
    #         cross_entropy_lst.append(np.log(1-p))
    # return -sum(cross_entropy_lst)
    
    return -sum([np.log(p) if y == 1 else np.log(1-p) for y, p in zip(Y, P)])


def cross_entropy_v2(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

Y = [1,0,1,1]
P = [0.4,0.6,0.1,0.5]

print(cross_entropy(Y, P))

print(cross_entropy_v2(Y, P))

# testing for cross_entropy_v2
Y = np.float_(Y)
P = np.float_(P)
y_one = Y * np.log(P)
y_zero = (1 - Y) * np.log(1 - P)
print(f'y_one: {y_one}')
print(f'y_zero: {y_zero}')
print(f'cross_list = {y_one + y_zero}')
print(f'cross_entro_list = {-np.sum(y_one + y_zero)}')