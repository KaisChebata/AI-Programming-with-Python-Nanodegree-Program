"""
In this Lab, we predict student admissions to graduate school at UCLA 
based on three pieces of data:
GRE Scores (Test)
GPA Scores (Grades)
Class rank (1-4)
"""

from initializer import data, np, pd

print(data.head())

# TODO: One-hot encoding the rank

# Use the get_dummies function in Pandas in order to one-hot encode the data.

# TODO:  Make dummy variables for rank
ranks_dummies = pd.get_dummies(data['rank'], prefix='rank')

one_hot_data = pd.concat([data, ranks_dummies], axis=1)
# print(ranks_dummies.head())

# TODO: Drop the previous rank column
one_hot_data = one_hot_data.drop('rank', axis=1)
# print(one_hot_data.head())
####################################################################################

# TODO: Scaling the data, Let's fit our two features into a range of 0-1
# Making a copy of our data
processed_data = one_hot_data[:]
processed_data['gre'] = processed_data['gre'] / 800
processed_data['gpa'] = processed_data['gre'] / 4.0
# print(processed_data.head())
####################################################################################

# Splitting the data into Training and Testing
# In order to test our algorithm, 
# we'll split the data into a Training and a Testing set. 
# The size of the testing set will be 10% of the total data.
sample = np.random.choice(
    processed_data.index, size=int(len(processed_data)*0.9), replace=False
    )
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

# print(train_data.head())
# print(test_data.head())
####################################################################################

# Splitting the data into features and targets (labels)

# Now, as a final step before the training, 
# we'll split the data into features (X) and targets (y).
features = np.float_(train_data.drop('admit', axis=1))
targets = np.float_(train_data['admit'])
features_test = np.float_(test_data.drop('admit', axis=1))
targets_test = np.float_(test_data['admit'])

# print(features[:10])
# print(targets[:10])
####################################################################################

# Training the 2-layer Neural Network
# The following function trains the 2-layer neural network. 
# First, we'll write some helper functions.

# Activation (sigmoid) function
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# sigmoid prime
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))

# error formula
def error_formula(y, output):
    return - (y * np.log(output) - (1 - y) * np.log(1-output))

# TODO: Backpropagate the error
# Write the error term. 
# Remember that this is given by the equation: -(y - y^)*simoid'(x)
# TODO: Write the error term formula
def error_term_formula(x, y, output):
    return (y - output) * sigmoid_prime(x)

# Training functions
def training_nn(features, targets, epochs, learnrate):
    
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1/ n_features**.5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            # Loop through all records, x is the input, y is the target

            # Activation of the output unit
            # Notice we multiply the inputs and the weights here 
            # rather than storing h as a separate variable
            output = sigmoid(np.dot(x, weights))

            # The error, the target minus the network output
            error = error_formula(y, output)

            # The error term
            error_term = error_term_formula(x, y, output)

            # The gradient descent step, the error times the gradient times the inputs
            del_w += error_term * x

        # Update the weights here. The learning rate times the 
        # change in weights, divided by the number of records to average
        weights += learnrate * del_w / n_records

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)

            print(f'Epoch: {e}')
            if last_loss and last_loss < loss:
                print(f'Train loss: {loss} - Warning - Loss Increasing')
            else:
                print(f'Train loss: {loss}')
            last_loss = loss
            print('============================')
    print('Finishing Training!')

    return weights

if __name__ == '__main__':
    # Neural Network hyperparameters
    epochs = 1000
    learnrate = 0.5

    print('Training ....')
    weights = training_nn(features, targets, epochs, learnrate)

    print('============================')
    
    # Calculating the Accuracy on the Test Data
    test_out = sigmoid(np.dot(features_test, weights))
    predictions = test_out > 0.5
    accuracy = np.mean(predictions == targets_test)
    # print("Prediction accuracy: {:.3f}".format(accuracy))
    print(f'Prediction accuracy: {accuracy:.3f}')