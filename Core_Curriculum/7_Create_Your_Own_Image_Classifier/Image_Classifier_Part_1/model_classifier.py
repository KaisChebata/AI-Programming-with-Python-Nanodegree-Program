import torch
from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_sizes=[], dropp=0.2):
        '''Feedforward Classifier (Fully-Connected Network) with arbitrary hidden layers units.
        
            Arguments
            ---------
            in_features: (int) Size of the input layer
            out_features: (int) Size of the output layer
            hidden_layers: (list of int) list of hidden units, the sizes of the hidden layers
        '''
        super().__init__()
        
        self.hidden_layers_sizes = hidden_layers_sizes
        self.dropp = dropp
        
        # check if user make choice of zero hidden layers
        if not self.hidden_layers_sizes:
            # first layer is the last layer and it is the output layer(only in and out)
            self.layers = nn.ModuleList([nn.Linear(in_features, out_features)])
        
        else:
            # in case there are at least one or more hidden layers
            self.layers = nn.ModuleList([nn.Linear(in_features, self.hidden_layers_sizes[0])])
            self.layers.extend(Classifier._hidden_layers_init(self.hidden_layers_sizes))
            self.output = nn.Linear(hidden_layers_sizes[-1], out_features)
        
        # adding dropout to reduce overfitting
        self.dropout = nn.Dropout(dropp)
    
    def forward(self, in_features):
        '''Forward Pass through the Fully-Connected Classifier 
            
            returns the output logits
        '''
        # check if no hidden layers, then output taken and returned through first layer
        if not self.hidden_layers_sizes:
            for layer in self.layers:
                in_features = layer(in_features)
            return F.log_softmax(in_features, dim=1)
        
        # in case there are more than one layer in the classifier, then output taken and returned through last layer
        else:
            for layer in self.layers:
                in_features = F.relu(layer(in_features))
                in_features = self.dropout(in_features)
        
            in_features = self.output(in_features)
        
        return F.log_softmax(in_features, dim=1)

    @staticmethod
    def _hidden_layers_init(hidden_sizes):
        '''Helper Function that detect and shape hidden layers based on layers size 
           to be added to the classifier ModuleList later'''
        hidden_layers_units = zip(hidden_sizes[:-1], hidden_sizes[1:])

        return nn.ModuleList([nn.Linear(*layer) for layer in hidden_layers_units])

if __name__ == '__main__':
    # test
    classifier_params = [1024, 102, [512, 256, 128, 64]]
    classifier_test = Classifier(*classifier_params)
    print(classifier_test)
    inputs = torch.rand(64, 1024)
    out_test = classifier_test(inputs)
    print(out_test.shape)

"""
# Hidden Layers number:
# Number of Hidden Layers and what they can achieve:

# 0 - Only capable of representing linear separable functions or decisions.

# 1 - Can approximate any function that contains a continuous mapping from 
#     one finite space to another.

# 2 - Can represent an arbitrary decision boundary to arbitrary accuracy 
#     with rational activation functions and can approximate any smooth mapping 
#     to any accuracy.

# neurons in hidden layer (hidden layer size):
# 1- The number of hidden neurons should be between the size of the input layer 
#    and the size of the output layer.
# 2- For a three layer network with n input and m output neurons, 
#   the hidden layer would have sqrt{n*m} neurons.
# 3- The number of hidden neurons should be power of 2 i.e. 16, 32, 64, 128 and so on.
"""

"""
# Setting learning rate:
# 1- in general, smaller learning rates will require more training epochs. 
# 2- Conversely, larger learning rates will require fewer training epochs. 
# 3- Further, smaller batch sizes are better suited to smaller learning rates 
#    given the noisy estimate of the error gradient.

# A traditional default value for the learning rate is 0.1 or 0.01, 
# and this may represent a good starting point on your problem.
"""