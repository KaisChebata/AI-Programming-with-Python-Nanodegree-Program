import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Some helper functions for plotting and drawing lines
def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]

    # admitted = np.array(X[np.argwhere(y==1)]).reshape(50, 2)
    # rejected = np.array(X[np.argwhere(y==0)]).reshape(50, 2)

    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], color = 'red', edgecolor = 'k')

    # plt.scatter(X[:, 0], X[:, 1], c=['r' if y==1 else 'b' for y in y], edgecolors='k', s=25)

    # plt.scatter(admitted[:, 0], admitted[:, 1], c='red', edgecolors='k')
    # plt.scatter(rejected[:, 0], rejected[:, 1], c='blue', edgecolors='k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)

def data_prep():
    data_fn = 'Lesson_1_Introduction_to_Neural_Networks/gradient_descent/data_gradient_descent.csv'
    data = pd.read_csv(data_fn, header=None)
    X = np.array(data[[0, 1]])
    y = np.array(data[2])
    return X, y

"""
TODO: Implementing the basic functions

Here is your turn to shine. 
Implement the following formulas, as explained in the text.

Sigmoid activation function:

σ(x)= 1 / 1 + e^−x

Output (prediction) formula:

y^=σ(w1x1+w2x2+b)

Error function:

Error(y,y^)=−ylog⁡(y^)−(1−y)log⁡(1−y^)

The function that updates the weights:

wi ⟶ wi + α(y−y^)xi

b ⟶ b + α(y−y^)
"""

# Implement the following functions

# Activation (sigmoid) function
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.matmul(features, weights) + bias)


# Error (log-loss) formula
def error_formula(y, output):
    return (-y * np.log(output) - (1 - y) * np.log(1 - output))

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = -(y - output)
    weights -= learnrate * d_error * x
    bias -= learnrate * d_error

    return weights, bias

def train(features, targets, epochs, learnrate, graph_lines=False):
    errors = []
    n_records, n_features = features.shape
    last_loss = None
    weights = np.random.normal(scale=1/ n_features**.5, size=n_features)
    bias = 0

    for e in range(epochs):
        for x, y in zip(features, targets):
            output = output_formula(x, weights, bias)
            error = error_formula(y, output)
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        
        # printing out log-loss error on training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)

        if e % (epochs / 10) == 0:
            print(f'\n============= Epoch {e} =============')
            if last_loss and last_loss < loss:
                print(f'Train loss: {loss} WARNING - Loss Increasing!!')
            else:
                print(f'Train loss: {loss}')
            
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print(f'Accuracy: {accuracy}')
        
        # the slope and y-intercept of the line:
        # weights[0]*x + weights[1]*y + bias = 0
        # -> y = -weights[0]/weights[1]*x - bias/weights[1]
        # -> slope = - weights[0]/weights[1]
        # -> intercept = - bias/weights[1]
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
    
    # Plotting the data
    plot_points(features, targets)

    # Plotting the solution boundary
    plt.title('Solution Boundary')
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()

if __name__ == '__main__':
    np.random.seed(44)
    epochs = 100
    learnrate = 0.01

    X, y = data_prep()
    
    train(X, y, epochs, learnrate, True)

    # some tests
    # features, data_num, weights, bias = train(X, y, epochs, learnrate, True)
    # print(f'features: {features}\nrecords_num: {data_num}\nx: {X[0]}\nweights: {weights}\nbais: {bias}')
    # print(f'weights shape: {weights.shape}')
    # print(f'feature(x) shape: {X[0].shape}')
    # # print(X)
    # print(f'X shape: {X.shape}')
    # print(f'weights shape: {weights.shape}')
    # print(1/features**.5)
    # w = np.random.normal(scale=1/ features**.5, size=features)
    # print(f'w size: {w.size}\nw dim: {w.ndim}\nw shape: {w.shape}')
    # print(np.matmul(X, weights))
    
    # w = np.random.normal(scale=1/X.shape[1]**.5, size=X.shape[1])
    # out = output_formula(X, w, np.random.randint(1, 10))
    # print(w)
    # print(out)
    # prediction = out > 0.5
    # print(prediction)
    # print(prediction == y)
    # print(np.mean(prediction == y))