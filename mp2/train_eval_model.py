"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression
from math import ceil

x_std = None
x_max = None
x_min = None


def normalise(data, mode):
    """
    Normalise the data to prevent overflow and reduce computation efforts
    Min-Max normalisation
    Must be used in training stage before called in evalution!!!
    Args:
        data(list): Data loaded from io_tools
        mode(str): 'train' for training stage, 'eval' in evaluation stage
    Returns:
        None
    """
    global x_std, x_max, x_min
    #
    if mode == 'train':
        # print ("Normalising data in training stage.")
        x_std = np.std(data[0], axis=0)
        x_max = np.amax(data[0], axis=0)
        x_min = np.amin(data[0], axis=0)
    elif mode == 'eval':
        # print("Normalising data in evalution stage.")
        pass
    else:
        print("Exception in normalising data, type: {0}".format(mode))
    data[0] = 1. * (data[0] - x_min) / (x_max - x_min)
    # data[1] = (data[1] - y_mean) / y_std
    pass


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    #
    def shuffle_data(data):
        """
        Shuffle the data, row mojor.

        Args:
            data(list): Unshuffled data [x y]
        Returns:
            shuffled(list): Shuffled version of input data
        """
        # Generate shuffled indices
        p = np.random.permutation(len(data[1]))
        shuffled = [data[0][p], data[1][p]]
        return shuffled
    #
    normalise(processed_dataset, mode='train')
    # Calcuate number of batches to complete an epoch,
    # last batch may be smaller.
    n_slice = int(ceil(len(processed_dataset[1]) / batch_size))
    # print ("Total number of steps to run: {0}".format(num_steps))
    #
    for step in range(num_steps):
        print("Step {:7d}/{:7d} | ".format(step + 1, num_steps), end='')
        # Shuffle at each epoch
        if shuffle and step % n_slice == 0:
            data = shuffle_data(processed_dataset)
        else:
            data = processed_dataset
        # Generate batch slice indices
        idx_beg = (step % n_slice) * batch_size
        idx_end = ((step + 1) % n_slice) * batch_size \
            if (step % n_slice) != (n_slice - 1) else None
        # print ("begin {0} end {1}".format(idx_beg, idx_end))
        #
        # slice the mini_batch from the data
        batch_x = data[0][idx_beg:idx_end, :]
        batch_y = data[1][idx_beg:idx_end]
        #
        update_step(batch_x, batch_y, model, learning_rate)
    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # forward
    f = model.forward(x=x_batch)
    # backward calculate gradient and loss
    grad = model.backward(f, y=y_batch)
    loss = model.total_loss(f, y=y_batch)
    print("Loss: {0}".format(loss))
    # update the weights
    n_data = y_batch.shape[0]
    model.w += -learning_rate * grad / n_data
    pass


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    # normalisation
    normalise(processed_dataset, mode='train')
    # _x = [x 1]
    n_data = processed_dataset[0].shape[0]
    _x = np.concatenate((processed_dataset[0], np.ones((n_data, 1))), axis=1)
    # closed form solution
    # w = (x.T * x + lambda * I) ^ -1 * x.T * y
    regularize = model.w_decay_factor * np.identity(model.ndims + 1)
    pseudo_inv = np.linalg.pinv(_x.T.dot(_x) + regularize)
    model.w = pseudo_inv.dot(_x.T).dot(processed_dataset[1])
    pass


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    n_eval = processed_dataset[1].shape[0]
    normalise(processed_dataset, mode='eval')
    pred = model.forward(x=processed_dataset[0])
    """
    # show all prediction result and corresponding losses
    for itr_pred, itr_label in zip(pred, processed_dataset[1]):
        error = 100. * (itr_pred[0] - itr_label[0]) / itr_label[0]
        print("GT: {:6d} | Pred: {:6d} | err: {:4.3f}%"
            .format(itr_label[0], int(itr_pred[0]), error))
    #
    """
    loss = model.total_loss(f=pred, y=processed_dataset[1])
    return loss
