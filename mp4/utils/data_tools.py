"""
Implements feature extraction and other data processing helpers.
"""

import numpy as np
import skimage
from skimage import color


def preprocess_data(data, process_method='default'):
    """Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1] by dividing by 255.
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 8*8*3)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Convert from rgb to gray then back to rgb. Use skimage
          3. Take the absolute value of the difference with the original image.
          4. Remove the global mean image
          5. Flatten images, data['image'] is converted to dimension (N, 8*8*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """

    def flatten_images(data):
        """
        """
        # Flatten images
        per_image = np.prod(data['image'].shape[1:])
        flatten_shape = [data['image'].shape[0], per_image]
        data['image'] = np.reshape(data['image'], flatten_shape)
        return data

    print("Preprocessing method: {0}".format(process_method))
    #
    if process_method == 'raw':
        # Convert the images to range of [0, 1]
        data['image'] = data['image'] / 255.
        # Remove mean.
        data = remove_data_mean(data)
        # Flatten images
        data = flatten_images(data)
        pass

    elif process_method == 'default':
        # Convert the images to range of [0, 1]
        data['image'] = data['image'] / 255.
        # Convert from rgb to gray then back to rgb. Use skimage
        gray_sc = color.rgb2gray(data['image'])
        rgb_img = color.gray2rgb(gray_sc)
        # Take the absolute value of the difference with the original image
        data['image'] = np.abs(data['image'] - rgb_img)
        # Remove the global mean image
        data = remove_data_mean(data)
        # Flatten images
        data = flatten_images(data)
        pass

    elif process_method == 'custom':
        # Design your own feature!
        pass
    return data


def compute_image_mean(data):
    """ Computes mean image.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        image_mean(numpy.ndarray): Average across the example dimension.
    """
    # calculate mean image (3 channels) across the entire dataset
    image_mean = data['image'].mean(axis=0)
    return image_mean


def remove_data_mean(data):
    """Removes data mean.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    # calculate mean across the entire dataset
    image_mean = compute_image_mean(data)
    # remove the mean from the data
    data['image'] = data['image'] - image_mean
    return data
