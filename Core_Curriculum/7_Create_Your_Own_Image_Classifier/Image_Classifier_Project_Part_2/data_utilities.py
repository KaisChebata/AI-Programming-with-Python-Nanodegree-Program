import os

import torch
from torchvision import datasets, transforms

def dataset_processor(data_dir):
    # get directory component of a pathname, then append it to directory
    base = os.path.abspath(os.path.dirname(data_dir))
    data_dir = os.path.join(base, data_dir)
    
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # apply data augmentation and normalization to train, valid, test data
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

    try:
        # Data loading
        image_datasets = {
            'train': datasets.ImageFolder(train_dir, transform=datat_transforms['train']), 
            'valid': datasets.ImageFolder(valid_dir, transform=datat_transforms['valid']), 
            'test': datasets.ImageFolder(test_dir, transform=datat_transforms['valid'])
        }

        # Data batching
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True), 
            'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True), 
            'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
        }
    except:
        print(
            'the path of data directory you pass does not exist'
            ' make sure you enter the correct data dir path')
        exit('Program exits ...')
    
    class_to_idx = image_datasets['train'].class_to_idx

    return dataloaders, class_to_idx
