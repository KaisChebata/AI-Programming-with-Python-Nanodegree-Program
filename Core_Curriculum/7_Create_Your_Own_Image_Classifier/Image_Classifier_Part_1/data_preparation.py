import os

import torch
from torchvision import datasets, transforms

base = os.path.abspath(os.path.dirname(__file__))
data_dir = os.path.join(base, 'flower_data')
# data_dir = 'flower_data'
train_dir = os.path.join(data_dir, 'train')
valid_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')

datat_transforms = {
    'train': transforms.Compose([transforms.RandomRotation(30), 
                                 transforms.RandomResizedCrop(224), 
                                 transforms.RandomHorizontalFlip(), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])]), 
    'valid': transforms.Compose([transforms.Resize(256), 
                                 transforms.CenterCrop(224), 
                                 transforms.ToTensor(), 
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=datat_transforms['train']), 
    'valid': datasets.ImageFolder(valid_dir, transform=datat_transforms['valid']), 
    'test': datasets.ImageFolder(test_dir, transform=datat_transforms['valid'])
}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True), 
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True), 
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
}


class_to_idx = image_datasets['train'].class_to_idx

