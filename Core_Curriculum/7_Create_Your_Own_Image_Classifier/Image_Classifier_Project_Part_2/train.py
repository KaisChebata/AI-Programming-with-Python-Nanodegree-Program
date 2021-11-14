from train_utilities import validation, train_model
from data_utilities import dataset_processor
from model_utilities import model_initializer, model_saver
from arguments_processor import get_training_args

# get args from command-line
in_args = get_training_args()

data_dir = in_args.data_directory
save_dir = in_args.save_directory
dropout = in_args.dropout
arch = in_args.arch.lower()
learnrate = in_args.learnrate
hidden_units = in_args.hidden_units
epochs = in_args.epochs
gpu = in_args.gpu

# get data set ready to be feeded to network
dataloaders, class_to_idx = dataset_processor(data_dir)

# get number of classes
out_features = len(class_to_idx)

# initializing the network
print('\nStart Initializing Network ....\n')

init_results = model_initializer(arch, out_features, hidden_units=hidden_units, 
                                drop_out=dropout, learnrate=learnrate)

# unpack initialization network results
arch, model, criterion, optimizer, learnrate, *classifier = init_results

print(f'Computation mode: {"GPU ENABLED" if gpu else "CPU ENABLED"}')

print('hyperparameters:')
print(
    f'arch: {arch} - learning rate: {learnrate} - hidden units: {hidden_units}'
    f' - epochs: {epochs} - dropout: {dropout}'
    )

# training
print('\nStart Training .......\n')

training_results = train_model(model, criterion, optimizer, 
                    dataloaders, epochs=epochs, gpu=gpu)

# unpack training results
trained_model = training_results['trained_model']
trained_optimizer = training_results['trained_optimizer']
training_losses = training_results['training_losses']
validating_losses = training_results['validating_losses']
accuracy_progress = training_results['accuracy_progress']
epochs = training_results['epochs']

# try save
print('Saving the trained model ....')
check_path = model_saver(save_dir, arch, trained_model, classifier, 
                        trained_optimizer, learnrate, class_to_idx, 
                        training_losses, validating_losses, 
                        accuracy_progress, epochs)

print('\nSaving Done!')
print(f'model had been saved at:\n{check_path}')

# Testing:
# train.py script tested on the following command-line:
# python train.py flowers --save_dir checkpoints_dir --arch resnet101 --learning_rate 0.001 --hidden_units 512 epochs 10 --gpu

# hyperparameters:
# arch: resnet101 - learning rate: 0.001 - hidden units: [512] - epochs: 10 - dropout: 0.2
# Training results:
