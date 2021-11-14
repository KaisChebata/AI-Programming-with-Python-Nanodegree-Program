import json
import matplotlib.pyplot as plt

from train import train_model, validation, testing
from model_utilities import model_initializer, model_saver, model_loader
from data_preparation import image_datasets, dataloaders
from images_utilities import imshow, process_image
# from model_classifier import Classifier

# with open('cat_to_name.json', 'r') as f:
#     cat_to_name = json.load(f)

# class_to_idx = image_datasets['train'].class_to_idx

# out_features = len(cat_to_name)

# intialization_results = model_initializer(
#     'vgg19', out_features, [1024, 512], learnrate=0.001
#     )

# arch, model, criterion, optimizer, learnrate, device = intialization_results

# print(model)

# training_results = train_model(model, criterion, optimizer, dataloaders, epochs=15)

# trained_model, trained_optimizer, training_losses, validating_losses, all_accuracy, epochs = training_results

# trying to understand the results
# print(f'Last Train Loss: {training_losses[-1]}')
# print(f'Last Validation Loss: {validating_losses[-1]}')
# print(f'Last Validation Accuracy: {all_accuracy[-1]}')

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 2, 1)
# plt.plot(training_losses, label='Training loss')
# plt.plot(validating_losses, label='Validation loss')
# plt.legend(frameon=False)
# plt.title('Loss Progress')

# plt.subplot(1, 2, 2)
# plt.plot(all_accuracy, label='Accuracy')
# plt.legend(frameon=False)
# plt.title('Accuracy Progress')
# plt.show()

# # Testing the network
# testing_results = testing(trained_model, dataloaders['test'], criterion)

# testing_loss, accuracy = testing_results
# print(f'Overall Testing Loss: {testing_loss}')
# print(f'Overall Accuracy Loss: {accuracy}')

# Saving trained model
# model_saver(arch, trained_model, trained_optimizer, learnrate, class_to_idx, training_losses, validating_losses, all_accuracy, epochs)


# image_path = 'flower_data/test/1/image_06743.jpg'
# img = process_image(image_path)
# imshow(img)
# print(img.shape)
# plt.show()
