"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = None
max_iters = None


def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    A, T = read_dataset_tf('../data/trainset', 'indexing.txt')
    # Initialize model.
    model = LogisticModel_TF(ndims=16, W_init='zeros')
    # Build TensorFlow training graph
    model.build_graph(learn_rate=0.0001)
    # Train model via gradient descent.
    score = model.fit(Y_true=T, X=A, max_iters=100000)
    print("Overall accuracy: \n{0}".format(score))
    print("{0}".format(score.shape))
    # Compute classification accuracy based on the return of the "fit" method
    pass


if __name__ == '__main__':
    tf.app.run()
