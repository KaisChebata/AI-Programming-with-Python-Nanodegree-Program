import numpy as np
import matplotlib.pyplot as plt

from images_utilities import process_image, imshow

def sanity_checking(img_path, probs, classes, names_dict):
    
    # get tensor image for model
    img = process_image(img_path)
    
    # get image name
    img_name = names_dict[img_path.split('/')[-2]]
    
    # get actual flowers names
    flowers_names = [names_dict[id] for id in classes]
    
    # set up figure to plot
    plt.figure(figsize=(5, 10))
    
    # ploting
    
    # plot flower iamge
    plt.subplot(2, 1, 1)

    ax1 = plt.gca()
    ax1.axis('off')
    imshow(img, ax1)
    plt.title(img_name)
    
    
    # plot bars fig
    plt.subplot(2, 1, 2)
    
    bars_coordinates = np.arange(len(probs))
    plt.barh(bars_coordinates, probs, align='center', )
    y_pos = np.arange(len(flowers_names))
    plt.yticks(y_pos, flowers_names)
    ax2 = plt.gca()
    ax2.invert_yaxis()