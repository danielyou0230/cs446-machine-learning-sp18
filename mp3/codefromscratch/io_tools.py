"""Input and output helpers to load in data.
"""
import numpy as np


def read_dataset(path_to_dataset_folder, index_filename):
    """ Read dataset into numpy arrays with preprocessing included
    Args:
        path_to_dataset_folder(str): path to the folder containing samples
                                     and indexing.txt
        index_filename(str): indexing.txt
    Returns:
        A(numpy.ndarray): sample feature matrix A = [[1, x1],
                                                     [1, x2],
                                                     [1, x3],
                                                     .......]
                                where xi is the 16-dimensional feature of
                                each sample

        T(numpy.ndarray): class label vector T = [y1, y2, y3, ...]
                             where yi is +1/-1, the label of each sample
    """
    ###############################################################
    # Fill your code in this function
    ###############################################################
    A = list()
    T = list()
    # Hint: open(path_to_dataset_folder+'/'+index_filename,'r')
    with open(path_to_dataset_folder + '/' + index_filename, 'r') as f:
        for line in f:
            # [label] [path_to_file]
            tmp_label, tmp_file = line.split()
            # convert str to int and append label to the array
            T.append(int(tmp_label))
            # load data from the file [path_to_file]
            with open(path_to_dataset_folder + '/' + tmp_file) as file_itr:
                # convert to float
                data = [float(itr) for itr in file_itr.readline().split()]
                # add bias at front and append data to the array
                A.append([1.] + data)
    #
    return np.array(A), np.array(T)
