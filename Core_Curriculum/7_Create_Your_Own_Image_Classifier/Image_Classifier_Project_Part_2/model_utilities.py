import os
import torch
from torchvision import models
from torch import nn, optim
from torchvision import models

from classifier import Classifier

# model initializer
def model_initializer(arch, out_features, hidden_units=[], drop_out=0.2, learnrate=0.001):
    '''Function that initialize the selected arch to be trained later.
    
            Arguments
            ---------
            arch: (str) the name of pre-trained model.
            out_features: (int) the number of class that model will predict.
            hidden_units: (list of int) list of hidden units for each hidden layer.
            learnrate: (float) the learning rate for the training process
            
            Returns
            -------
            arch: (str) the name of pre-trained model.
            model: (torchvisoin.models) pre-trained model initialized with new classifier.
            criterion: (nn.NLLLoss()) negative log likelihood loss.
            optimizer: (torch.optim) initialized optimizer for update weights matrices.
            device: (torch.device) the available device to work on.
            
    '''
    
    #used models are: vgg19, alexnet, resnet101
    archs_dict = {'vgg19': models.vgg19, 
                  'alexnet': models.alexnet, 
                  'resnet101': models.resnet101
                 }
    
    # load the selected model from torchvision.models
    try:
        model = archs_dict[arch](pretrained=True)
    except KeyError:
        print(
            f'the architecuture that you set, does not supported for this App.\n'
            f'only the following architecutures are supported: '
            f'{", ".join([k for k in archs_dict.keys()])}'
            )
        exit('Program exits ...')
    
    # getting input features of the original arch's classifier
    model_classifier = model.classifier if hasattr(model, 'classifier') else model.fc
    in_features = (
        model_classifier[0].in_features if isinstance(model_classifier, nn.Sequential) 
        else model_classifier.in_features
    )
    
    # Creating new FeedForward classifier to bind and train it with loaded model
    classifier = Classifier(in_features, out_features, hidden_layers_sizes=hidden_units, dropp=drop_out)
    
    # Freeze model's parameter to avoid updating them during backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # attach the new classifier to the loaded model
    if hasattr(model, 'classifier'):
        model.classifier = classifier
    else:
        model.fc = classifier
    
    # define criterion
    criterion = nn.NLLLoss()
    
    # ** Train the classifier parameters
    optimizer = optim.Adam(
        model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters(), 
        lr=learnrate)
    
    return (arch, model, criterion, optimizer, learnrate,  
            in_features, out_features, hidden_units, drop_out)


def model_saver(save_dir, arch, model, classifier, optimizer, learnrate, 
                class_to_idx, training_losses, validating_losses, 
                accuracy_progress, epochs):
    
    # construct checkpoint dict
    checkpoint = {'arch': arch, 
                  'classifier': classifier,
                  'state_dict': model.state_dict(), 
                  'optimizer_state_dict': optimizer.state_dict(), 
                  'learnrate': learnrate, 
                  'class_to_idx': class_to_idx, 
                  'training_losses': training_losses, 
                  'validating_losses': validating_losses, 
                  'accuracy_progress': accuracy_progress, 
                  'training_loss': training_losses[-1], 
                  'validation_loss': validating_losses[-1], 
                  'validation_accuracy': accuracy_progress[-1],
                  'epochs': epochs
                  }
    
    # make the arch name prefix the checkpoint name when saving it
    try:
        base = os.path.abspath(os.path.dirname(save_dir))
        save_dir = os.path.join(base, save_dir)
        checkpoint_path = os.path.join(base, save_dir, f'{arch}_checkpoint.pth')
        
        torch.save(checkpoint, checkpoint_path)
    except:
        checkpoint_path = os.path.join(f'{arch}_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)

        print(
            f'the path of save directory you pass does not correct.'
            f' make sure you enter the correct save dir path.\n'
            f'Because the valuable training efforts and GPU'
            f' cost computaion we save the checkpoint at:\n./{checkpoint_path}'
            )
        exit('Program exits ...')
    
    return checkpoint_path

# TODO: Write a function that loads a checkpoint and rebuilds the model

def model_loader(filepath):
    
    # loads checkpoint
    checkpoint= torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # rebuild the model with appropriate arch and classifier
    
    # loads the arch
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    
    # freeze early parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # loads classifier arguments and create it from the checkpoint
    classifier = Classifier(*checkpoint['classifier'])
    
    # attach the classifier to the model
    if hasattr(model, 'classifier'):
        model.classifier = classifier
    else:
        model.fc = classifier
    
    # loads the state dict of trained model from checkpoint into new model
    model.load_state_dict(checkpoint['state_dict'])
    
    # attach class_to_idx map to the new model
    model.class_to_idx = checkpoint['class_to_idx']

    # define an new optimizer with learning rate and update it with saved optimizer_state_dict
    # for continue training later.
    learnrate = checkpoint['learnrate']
    optimizer = optim.Adam(
        model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters(), 
        lr=learnrate)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer