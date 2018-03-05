"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers
from math import ceil


def train_model(data, model, learning_rate=0.001, batch_size=64,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    def suffle_data(data):
        """ Shuffle the given data.
            Args:
                data(dict): [X, Y] input data with input X and label Y.
            Returns:
                shuffled(dict): [X, Y] shuffled version of the input data.
        """
        indices = np.random.permutation(len(data['label']))
        shuffled = {'image': data['image'][indices],
                    'label': data['label'][indices]}
        return shuffled

    test_count = 200
    history = list()
    # Calculate the number of slice per epoch
    n_instance = len(data['label']) - test_count
    n_slice = int(ceil(n_instance / batch_size))
    print(" - Total slices: {0}".format(n_slice))
    # Processing data
    # Shuffle the entire dataset to give them a great mix
    shuffled = suffle_data(data) if shuffle else data
    # Slice out the training data
    trainset = {'image': shuffled['image'][:-test_count, :],
                'label': shuffled['label'][:-test_count, :]}
    # Slice out the testing data
    testset = {'image': shuffled['image'][-test_count:, :],
               'label': shuffled['label'][-test_count:, :]}
    # Performs gradient descent.
    for step in range(num_steps):
        print("Step: {:5d}/{:5d}".format(step + 1, num_steps), end='')
        # Shuffle the training data every epoch
        if step > 0 and step % n_slice == 0:
            trainset = suffle_data(trainset)
        # mini-Batch
        # Generate batch slice indices
        idx_beg = (step % n_slice) * batch_size
        idx_end = ((step + 1) % n_slice) * batch_size \
            if (step % n_slice) != (n_slice - 1) else None
        # slice the mini-Batch
        batch_img = trainset['image'][idx_beg:idx_end, :]
        batch_lbl = trainset['label'][idx_beg:idx_end, :]
        #
        update_step(batch_img, batch_lbl, model, learning_rate)
        # self evaluation on training data
        minibatch = {'image': batch_img, 'label': batch_lbl}
        _, acc = eval_model(minibatch, model)
        print(" | Training Acc: {:3.2f}%".format(acc * 100.))
        # Test the model with testing data every 500 steps
        if step > 0 and step % 500 == 0:
            _, acc = eval_model(testset, model)
            # record the in-stage testing accuarcy
            history.append(acc * 100.)
    #
    # Final test of the model with testing data
    _, acc = eval_model(testset, model, show=False)
    history.append(acc * 100.)
    print(history)

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # forward
    f = model.forward(x_batch)
    # backward
    gradients = model.backward(f, y_batch)
    # calculate the loss
    loss = model.total_loss(f, y_batch)
    print(" | Loss: {:5.10f}".format(loss), end='')
    # update weights
    model.w = model.w - learning_rate * gradients
    pass


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Your implementation here (do not modify the code above)
    threshold = 1.0e-05
    z = np.array([itr[0] if itr[0] > threshold else 0. for itr in z])
    z = z[:, np.newaxis]
    # w = X.T
    # Set model.w
    bias = np.ones((data['label'].shape[0], 1))
    _x = np.concatenate((data['image'], bias), axis=1)
    model.w = _x.T.dot(z * data['label'])


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    n_instance = data['label'].shape[0]
    # P[i, j] = yi * yj * xi.T * xj
    bias = np.ones((data['label'].shape[0], 1))
    _x = np.concatenate((data['image'], bias), axis=1)
    x_matrix = _x.dot(_x.T)
    y_matrix = data['label'].dot(data['label'].T)
    P = y_matrix * x_matrix
    # q = [-1, -1, ..., -1].T
    q = -1. * np.ones((n_instance, 1))
    # G = [[I], [-I]]
    # constraint 1:  margin <= C
    # constraint 2: -margin <= 0
    constraint_1 = +1 * np.identity(n_instance)
    constraint_2 = -1 * np.identity(n_instance)
    G = np.vstack([constraint_1, constraint_2])
    # h = [[C], [0]]
    # C can be set as demand
    C = 100.
    print("C = {:.3f}".format(C))
    h_constraint_1 = C * np.ones((n_instance, 1))
    h_constraint_2 = np.zeros((n_instance, 1))
    h = np.vstack([h_constraint_1, h_constraint_2])
    return P, q, G, h


def eval_model(data, model, show=False):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """

    def cal_accuracy(y, pred):
        return np.average([+1 if itr_y == itr_p else 0
                           for itr_y, itr_p in zip(y, pred)])

    f = model.forward(data['image'])
    pred = model.predict(f)
    if show:
        print(f)
        for p, l in zip(pred, data['label']):
            print("pred: {:2d} | label: {:2d}".format(p, l[0]))
    #
    loss = model.total_loss(f, data['label'])
    acc = cal_accuracy(data['label'], pred)
    return loss, acc
