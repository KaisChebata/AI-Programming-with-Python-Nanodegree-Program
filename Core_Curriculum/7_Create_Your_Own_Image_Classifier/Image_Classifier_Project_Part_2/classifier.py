from torch import nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_sizes=[], dropp=0.2):
        '''Feedforward Classifier (Fully-Connected Network) 
            with arbitrary hidden layers units.
        
            Arguments
            ---------
            in_features: (int) Size of the input layer
            out_features: (int) Size of the output layer
            hidden_layers: (list of int) list of hidden units, 
            the sizes of the hidden layers
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
