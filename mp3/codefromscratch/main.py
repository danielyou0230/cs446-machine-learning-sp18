"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset', 'indexing.txt')
    A, T = read_dataset('../data/trainset', 'indexing.txt')
    # print("A shape: {0} | T shape:{1}".format(A.shape, T.shape))
    # Initialize model.
    model = LogisticModel(ndims=16, W_init='zeros')
    # Train model via gradient descent.
    model.fit(Y_true=T, X=A, learn_rate=0.0001, max_iters=50000)
    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')
    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')
    # Try all other methods: forward, backward, classify, compute accuracy

    pass
