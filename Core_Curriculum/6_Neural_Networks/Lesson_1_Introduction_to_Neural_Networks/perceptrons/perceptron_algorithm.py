import numpy as np
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    # print(f'X: {X}\nW: {W}\nb: {b}')
    # print(f'linear function: {(np.matmul(X,W)+b)}')
    # print(f'input to step function: {(np.matmul(X,W)+b)[0]}')
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i] - y_hat == 1:
            W[0] += X[i][0] * learn_rate
            W[1] += X[i][1] * learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:
            W[0] -= X[i][0] * learn_rate
            W[1] -= X[i][1] * learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for _ in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)

        # add lines in slope intercept form: y = mx + b
        # where slope = m, intercept = b
        # here the linear equation we have is W1*x1 + W2*x2 + b = 0
        # => W2*x2 = -W1*x1 - b
        # => x2 = -(W1/W2)*x1 - b/W2
        # => slope = -(W1/W2), intercept = -b/W1
        boundary_lines.append((-W[0]/W[1], -b/W[1]))

        # helper prints
        # print(f'epoch #{_} ->:\n')
        # print(f'{W}')
        # print(f'b: {b}')
        # print(f'W shape: {len(W)}')
    
    # returns: 
    # steps (lines): where last line is the solution
    # x_min, x_max: to set x-axis range
    # y_min, y_max: to set y-axis range
    return boundary_lines, [x_min, x_max], [y_min, y_max]

# helper function that plot the line based on slope and intercept
def line_ploter(slope, intercept, *args, **kwargs):
    axes = plt.gca() # get current axes
    x_values = np.array(axes.get_xlim()) # get xes vals
    # define the line in slope intercept
    y_values = slope * x_values + intercept
    axes.plot(x_values, y_values, *args, **kwargs)

if __name__ == '__main__':
    # get data from external source
    data_fname = 'Lesson_1_Introduction_to_Neural_Networks/perceptrons/data_perceptron_algorithm.csv'
    data = np.genfromtxt(data_fname, delimiter=',')

    # prepare data (points) to be plotted.
    X = data[:, 0:2] # points in form [x1 x2]
    x1, x2 = X[:, 0], X[:, 1]
    y = data[:, 2] # labels of points 0 or 1
    # define the color for the data set blue for positive label, red otherwise
    data_set_colors = ['b' if y == 1 else 'r' for y in y]
    
    # get & plot solution
    lines = trainPerceptronAlgorithm(X, y)
    plt.scatter(x1, x2, c=data_set_colors)
    ax = plt.gca()
    ax.set_xlim(lines[1])
    ax.set_ylim(lines[2])

    # draw the steps of the Alogrithm
    for i in range(len(lines[0])):
        line_ploter(lines[0][i][0], lines[0][i][1], *['--g'], **{'linewidth': 0.5})
    
    # plot the final solution as a black line that classify the dataset
    line_ploter(lines[0][-1][0], lines[0][-1][1], *['black'], **{'linewidth': 1})
    plt.show()
    
    
