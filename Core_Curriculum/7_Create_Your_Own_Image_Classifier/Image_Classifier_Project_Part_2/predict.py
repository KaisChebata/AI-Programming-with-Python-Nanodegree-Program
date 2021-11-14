import json

from predict_utilities import predict
from model_utilities import model_loader
from arguments_processor import get_predict_args

# get input args from command-line
in_args = get_predict_args()

image_path = in_args.image_path
checkpoint = in_args.checkpoint
topk = in_args.topk
category_names = in_args.category_names
gpu = in_args.gpu

flowers_names = None

print(f'Image path: {image_path}')
print(f'checkpoint used: {checkpoint}')
print(f'Top-K classes inferenced num: {topk}')

print(f'Computation mode: {"GPU ENABLED" if gpu else "CPU ENABLED"}\n')

# load the checkpoint model
checkpoint_model, _ = model_loader(checkpoint)

# make inference and get results: probs and classes
probs, classes = predict(image_path, checkpoint_model, topk=topk, gpu=gpu)

if category_names: 
    print('file used for mapping categories to real names')
    with open(category_names, 'r') as file:
        cat_to_name = json.load(file)
    
    flowers_names = [cat_to_name[id] for id in classes]

print('Results:\n')
print('inferenced Probabilities, inferenced Classes:')
print(probs)
print(classes)

if flowers_names:
    print('flowers names:')
    print(flowers_names)

