"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    data = {}
    arr_img = list()
    arr_lbl = list()
    with open(data_txt_file, 'r') as f:
        for line in f:
            tmp_file, tmp_label = line.split(',')
            tmp_image = io.imread(image_data_path + tmp_file + '.jpg')
            #
            arr_lbl.append(int(tmp_label))
            arr_img.append(tmp_image)
    #
    data['label'] = np.array(arr_lbl)[:, np.newaxis]
    data['image'] = np.array(arr_img)
    return data
