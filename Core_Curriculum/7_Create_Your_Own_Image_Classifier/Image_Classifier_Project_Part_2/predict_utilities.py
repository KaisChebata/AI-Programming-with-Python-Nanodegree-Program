import torch

from image_utilities import process_image

def predict(image_path, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using 
        a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Use GPU torch when available
    device = torch.device('cuda' if gpu else 'cpu')
    
    # move model to the available device
    model.to(device)
    
    # mapping indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # get tensor object(image)
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)
    
    # move image to available device
    image = image.to(device)
    
    # set model to eval mode
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        
        # get the probabilities
        ps = torch.exp(output)
    
        # get top 5 classes with highest probabilities
        probs, indices = ps.topk(topk, dim=1)
        
        probs = probs.tolist()[0]
        classes = [idx_to_class[idx] for idx in indices.tolist()[0]]
        
    return probs, classes