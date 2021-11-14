import torch

# function that will validate the model after each pass from trainingloader
def validation(model, validationloader, criterion, device):
    '''Function that validate the performance of a model 
       every epoch after each batch passed to the model
       
        Arguments
        ---------
        model: (torchvisoin.models) a model that in training phase.
        validationloader: (dataloader) data to be used for validate the after batch passed.
        criterion: (nn.NLLLoss()) negative log likelihood loss.
        
        Returns
        -------
        validation_loss: the total of Negative log likelihood loss for all samples provided.
        accuracy: the total accuracy for all samples provided.
    
    '''
    
    validation_loss, accuracy = 0, 0
    
    for inputs, labels in validationloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        logps = model(inputs)
        validation_loss += criterion(logps, labels).item()
        
        # get the probabilities
        ps = torch.exp(logps)
        
        # get top class with highest probability
        _, top_class = ps.topk(1, dim=1)
        
        # Calculate accuracy
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return validation_loss, accuracy

# function that will train the model

def train_model(model, criterion, optimizer, dataloaders, epochs=1, gpu=False):
    '''Function that will train the network of a model
       
        Arguments
        ---------
        model: (torchvisoin.models) a pre-trained arch model to trained.
        criterion: negative log likelihood loss.
        optimizer: (torch.optim) an optimizer for update weights matrices.
        dataloaders: (dict) dict of dataloaders to be used to feed the network in the training and validation phases.
        epochs: (int) number of epochs the model will be trained over the dataset.
        
        Returns
        -------
        model: (torchvisoin.models) pre-trained model with new trained classifier.
        optimizer: (torch.optim) trained optimizer for update weights matrices.
        training_losses: (list of floats) list of all training losses.
        validating_losses: (list of floats) list of all validation losses.
        accuracy_progress: (list of floats) list of all accuracy.
        epochs: number of epochs the network trained in.
        
    '''
    # Use GPU if it's available
    device = torch.device('cuda' if gpu else 'cpu')

    # put model in current available device
    model.to(device)

    # track the running (current) training loss for a training phase
    current_training_loss = 0
    
    # tracking each epoch the overall losses for both training and validation phases, 
    # and all accuracy for each epoch
    training_losses, validating_losses, accuracy_progress = [], [], []
    
    # tracking the phase i.e. training or validation
    phases = ['train', 'valid']
    
    print('Training Progress during each epoch ...\n')
    
    for epoch in range(epochs):
        
        for phase in phases:
            if phase == 'train':
                model.train()
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    logps = model(inputs)
                    loss = criterion(logps, labels)
                    
                    loss.backward()
                    optimizer.step()
                    
                    current_training_loss += loss.item()
                
            else:
                # set model to eval mode
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    validation_loss, accuracy = validation(model, dataloaders[phase], 
                                                            criterion, device)
        
        # Keep tracking for looses for both pahses, and tracking accuracy
        training_losses.append(current_training_loss/len(dataloaders['train']))
        validating_losses.append(validation_loss/len(dataloaders['valid']))
        accuracy_progress.append(accuracy/len(dataloaders['valid']))
        
        
        print(
            f'Epoch: {epoch + 1}/{epochs}..  '
            f'Train Loss: {current_training_loss/len(dataloaders["train"]):.3f}..  '
            f'Validation Loss: {validation_loss/len(dataloaders["valid"]):.3f}..  '
            f'Validation Accuracy: {accuracy/len(dataloaders["valid"]):.3f}..  '
        )
        current_training_loss = 0
    
    print('\nTraining Done!....')

    results = {
        'trained_model': model, 
        'trained_optimizer': optimizer, 
        'training_losses': training_losses, 
        'validating_losses': validating_losses, 
        'accuracy_progress': accuracy_progress, 
        'epochs': epochs
    }
    
    return results


