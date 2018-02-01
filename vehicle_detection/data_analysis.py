import numpy as np
import pickle
import cv2
import glob
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def explore_input_images(car_images, noncar_images, n, out_dir=None):
    """
    Helper function to explore the input data.
    """
    data_dict = {}
    
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_images)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(noncar_images)
    # Read in a test image, either car or notcar
    example_img = mpimg.imread(car_images[0])
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_img.dtype
    
    h = n//4
    w = n//4
    fig, axs = plt.subplots(h, w, figsize=(16, 16))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()

    for i in np.arange(n):
        img = cv2.imread(car_images[np.random.randint(0,len(car_images))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('car', fontsize=10)
        axs[i].imshow(img)
    if out_dir:
        plt.savefig()
    plt.show()
    
    fig, axs = plt.subplots(h, w, figsize=(16, 16))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    
    for j in np.arange(n):
        img = cv2.imread(noncar_images[np.random.randint(0,len(noncar_images))])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axs[j].axis('off')
        axs[j].set_title('non_car', fontsize=10)
        axs[j].imshow(img)
    if out_dir:
        plt.savefig()
    plt.show()
       
    # Return data_dict
    return data_dict

def plot_as_subplots(in_image, out_image, in_title, out_title, out_dir=None, cmap_in=None, cmap_out=None):
    """
    Helper function to plot an image processed image as subplot of input and output image to 
    visualize the effect of transform applied.
    """
    f, (axis1, axis2) = plt.subplots(1, 2, figsize=(8,8))
    axis1.imshow(in_image, cmap=cmap_in)
    axis1.set_title(in_title, fontsize=16)
    axis2.imshow(out_image, cmap=cmap_out)
    axis2.set_title(out_title, fontsize=16)
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig('{}/{}.jpg'.format(out_dir, out_title))
    plt.show()
