import numpy as np
from PIL import Image
import torch

def process_image(image):
    ''' Function that scales, crops, and normalizes a PIL image 
        to be feded into a PyTorch model.
        
        Arguments
        ---------
        image: (string | pathlike) image path for image to be processed.
        
        Returns
        -------
        returns an Tensor object - tensor image
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model

    # image = Image.open(image).resize((256, 256))
    image = Image.open(image, 'r')

    # image dims and aspect ratio
    width, height = image.size
    aspect_ratio = width / height

    #  keeping the aspect ratio resize, where the shortest side is 256 pixels
    
    # if width is shortest side make it 256, 
    # and resize height to be proportional with original aspect ratio
    width = 256 if aspect_ratio < 1 else int(256 * aspect_ratio)
    
    # if height is shortest side make it 256, 
    # and resize width to be proportional with original aspect ratio
    height = 256 if aspect_ratio > 1 else int(256 * (aspect_ratio ** -1))

    # resizing
    image = image.resize((width, height))
    
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    
    crop_width, crop_height = (224, 224)

    upper, left = (width - crop_width) // 2, (height - crop_height) // 2
    bottom, right = (width + crop_width) // 2, (height + crop_height) // 2
    crop_box = (upper, left, bottom, right) 
    croped_image = image.crop(crop_box)

    np_img = ((np.array(croped_image) / 255.0) - means) / stds
    
    tensor_img = torch.from_numpy(np_img.transpose(2, 0, 1))
    
    return tensor_img
