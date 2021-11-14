import argparse

def get_training_args():
    
    # a parser to collect the arguments
    parser = argparse.ArgumentParser(description='Get inputs as arguments from user')

    parser.add_argument('data_directory', action='store', 
                        help='Stands for the directory where the dataset is located' 
                        ' to be feeded into model during training.' 
                        )
    
    parser.add_argument('--save_dir', action='store', default='',  
                        dest='save_directory', 
                        help='Stands for the directory where the model will be saved' 
                        ' as a checkpoint after training complete.'
                        )

    parser.add_argument('--dropout', action='store', default=0.2,  
                        dest='dropout', type=float, 
                        help='regularization technique for' 
                        ' reducing overfitting in the networks during training.' 
                        )
    
    parser.add_argument('--arch', action='store', default='vgg19',  
                        dest='arch', 
                        help='Stands for the pre-trained architecuture to be used' 
                        ' with the classifier to train the network, e.g., vgg19.'
                        ' user can choose either vgg19, resnet101, or alexnet.'
                        )
    
    parser.add_argument('--learning_rate', action='store', default=0.0001,  
                        dest='learnrate', type=float, 
                        help='Stands for the learning rate of the network'
                        ' during the training.'
                        )
    parser.add_argument('--hidden_units', action='store', default=[512], 
                        dest='hidden_units', type=int, nargs='+',
                        help='Stands for the hidden layers number'
                        ' and nodes in each of them -'
                        ' enter hidden units space separated, e.g, 512 256.'
                        )
                    
    parser.add_argument('--epochs', action='store', default=1, 
                        dest='epochs', type=int,
                        help='Stands for the number of epochs'
                        ' the model will train dure them.'
                        )

    parser.add_argument('--gpu', action='store_true', default=False, 
                        dest='gpu', 
                        help='Stands for boolean switch' 
                        ' to indicates Using GPU for training, Set a switch to True'
                        )
    return parser.parse_args()


def get_predict_args():
    
    # a parser to collect the arguments
    parser = argparse.ArgumentParser(description='Get inputs as arguments from user')

    parser.add_argument('image_path', action='store', 
                        help='Stands for the image path where it is located' 
                        ' for inference.' 
                        )
    
    parser.add_argument('checkpoint', action='store',  
                        help='Stands for the checkpoint path where it used to' 
                        ' loads the trained model for inference.'
                        )

    parser.add_argument('--top_k', action='store', default=5,  
                        dest='topk', type=int, 
                        help='Stands for the top k classes that will be predicted' 
                        ' in the inference process.' 
                        )
    
    parser.add_argument('--category_names', action='store',  
                        dest='category_names', 
                        help='Stands for categories file path that used for mapping'
                        ' from predicted encoded classes to real predicted names.'
                        )

    parser.add_argument('--gpu', action='store_true', default=False, 
                        dest='gpu', 
                        help='Stands for boolean switch' 
                        ' to indicates Using GPU for inference process, Set a switch to True'
                        )
    return parser.parse_args()

