import numpy as np
import cv2
import os

def calc_Normalize(images_paths: list):
    """
        images: list of image paths we want to calculate the values for
                e.g.: ['C:/Users/...']
    """
    # Load images
    images = []
    for filename in images_paths:
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)

    # Calc mean - - - - - -
    # - - > convert list of images to a 4D numpy array (num_images, height, width, channels)
    img_array = np.array(images)

    # Calculate the mean across the first axis (num_images)
    channel_means = np.mean(img_array, axis=(0, 1, 2))
    # Calculate the standard deviation across the first three axes (num_images, height, width)
    channel_stddevs = np.std(img_array, axis=(0, 1, 2))

    print(channel_means)
    print(channel_stddevs)
    return channel_means, channel_stddevs

# calc_Normalize(["C:/Users/klein/Desktop/BWKI/Final/images/1.png", 
#                 "C:/Users/klein/Desktop/BWKI/Final/images/2.png"])