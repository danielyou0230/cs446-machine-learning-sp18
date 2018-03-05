"""logistic model class for binary classification."""

import numpy as np
from math import ceil


class LogisticModel(object):
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of W is the bias term,
            self.W = [Bias, W1, W2, W3, ...]
            where Wi correspnds to each feature dimension

        W_init needs to support:
          'zeros': initialize self.W with all zeros.
          'ones': initialze self.W with all ones.
          'uniform': initialize self.W with uniform random number between [0,1)
          'gaussian': initialize self.W with gaussion distribution (0, 0.1)

        Args:
            ndims(int): feature dimension
            W_init(str): types of initialization.
        """
        self.ndims = ndims + 1
        self.W_init = W_init
        self.W = None
        ###############################################################
        # Fill your code below
        ###############################################################
        weight_shape = (self.ndims)
        if W_init == 'zeros':
            self.W = np.zeros(shape=weight_shape)
            pass
        elif W_init == 'ones':
            self.W = np.ones(shape=weight_shape)
            pass
        elif W_init == 'uniform':
            self.W = np.random.uniform(low=0.0, high=1.0, size=weight_shape)
            pass
        elif W_init == 'gaussian':
            self.W = np.random.normal(loc=0., scale=0.1, size=weight_shape)
            pass
        else:
            print('Unknown W_init ', W_init)

    def save_model(self, weight_file):
        """ Save well-trained weight into a binary file.
        Args:
            weight_file(str): binary file to save into.
        """
        self.W.astype('float32').tofile(weight_file)
        print('model saved to', weight_file)

    def load_model(self, weight_file):
        """ Load pretrained weghit from a binary file.
        Args:
            weight_file(str): binary file to load from.
        """
        self.W = np.fromfile(weight_file, dtype=np.float32)
        print('model loaded from', weight_file)

    def sigmoid(self, a):
            # return np.reciprocal(1 + np.exp(-a))
            return 1. / (1. + np.exp(-a))

    def cal_loss(self, X, Y):
        loss = np.log(1. + np.exp(-Y * (X.dot(self.W[:, np.newaxis]))))
        return np.sum(loss)

    def forward(self, X):
        """ Forward operation for logistic models.
            Performs the forward operation, and return probability
            score (sigmoid).
        Args:
            X(numpy.ndarray): input dataset with a dimension of
                            (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): probability score of (label == +1) for
                             each sample with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # probability = sigmoid(W.T * X)
        scores = self.sigmoid(X.dot(self.W[:, np.newaxis]))
        return scores

    def backward(self, Y_true, X):
        """ Backward operation for logistic models.
            Compute gradient according to the probability loss on
            lecture slides
        Args:
            X(numpy.ndarray): input dataset with a dimension of
                             (# of samples, ndims+1)
            Y_true(numpy.ndarray): dataset labels with a dimension of
                             (# of samples,)
        Returns:
            (numpy.ndarray): gradients of self.W
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        def cal_gradient(Y_true, X):
            _Y = Y_true[:, np.newaxis]
            nominator = -_Y * X * np.exp(-_Y * (X.dot(self.W[:, np.newaxis])))
            denominator = 1. + np.exp(-_Y * (X.dot(self.W[:, np.newaxis])))
            return np.sum(nominator / denominator, axis=0)
        #
        gradients = cal_gradient(Y_true, X)
        return gradients

    def classify(self, X):
        """ Performs binary classification on input dataset.
        Args:
            X(numpy.ndarray): input dataset with a dimension of
                              (# of samples, ndims+1)
        Returns:
            (numpy.ndarray): predicted label = +1/-1 for each sample
                             with a dimension of (# of samples,)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        logits = np.around(self.forward(X))
        pred = np.array([+1 if itr[0] == 1. else -1 for itr in logits])
        return pred

    def fit(self, Y_true, X, learn_rate, max_iters, batch_size=50):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of
                                   (# of samples,)
            X(numpy.ndarray): input dataset with a dimension of
                                   (# of samples, ndims+1)
            learn_rate: learning rate for gradient descent
            max_iters: maximal number of iterations
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        test_count = 200
        history = list()
        #
        print("Training Info:")
        print(" - Learning Rate  : {:f}".format(learn_rate))
        print(" - Batch Size     : {:d}".format(batch_size))
        print(" - Max Iterations : {:d}".format(max_iters))
        print(" - Testing Amount : {:d}".format(test_count))

        def suffle_data(data):
            """ Shuffle the given data.
            Args:
                data(list): [X, Y] input data with input X and label Y.
            Returns:
                shuffled(list): [X, Y] shuffled version of the input data.
            """
            indices = np.random.permutation(len(data[1]))
            shuffled = [data[0][indices], data[1][indices]]
            return shuffled
        # Shuffle the entire dataset to give them a great mix
        X, Y_true = suffle_data([X, Y_true])
        # Slice out the training data
        _x, _y = suffle_data([X[:-test_count, :], Y_true[:-test_count]])
        # Calculate the number of slice per epoch
        n_instance = len(Y_true) - test_count
        n_slice = int(ceil(n_instance / batch_size))
        print(" - Total slices: {0}".format(n_slice))
        # Slice out the testing data
        test_x = X[-test_count:, :]
        test_y = Y_true[-test_count:]
        # Train the model using mini-Batch SGD
        for step in range(max_iters):
            print("Step: {:5d}/{:5d}".format(step + 1, max_iters), end='')
            # Shuffle the training data every epoch
            if step > 0 and step % n_slice == 0:
                _x, _y = suffle_data([_x, _y])
            # mini-Batch
            # Generate batch slice indices
            idx_beg = (step % n_slice) * batch_size
            idx_end = ((step + 1) % n_slice) * batch_size \
                if (step % n_slice) != (n_slice - 1) else None
            # slice the mini-Batch
            batch_x = _x[idx_beg:idx_end, :]
            batch_y = _y[idx_beg:idx_end]
            # forward
            logits = self.forward(batch_x)
            # calculate the loss
            loss = self.cal_loss(batch_x, batch_y[:, np.newaxis])
            print(" | Loss: {:5.7f}".format(loss), end='')
            # backward
            gradients = self.backward(batch_y, batch_x)
            # update weights
            self.W = self.W - learn_rate * gradients
            # self evaluation on training data
            acc = self.eval(batch_x, batch_y, batch_size, show=False)
            print(" | Training Acc: {:3.2f}%".format(acc))
            # Test the model with testing data every 500 steps
            if step > 0 and step % 500 == 0:
                acc = self.eval(test_x, test_y, test_count)
                # record the in-stage testing accuarcy
                history.append(acc)
        # Final test of the model with testing data
        acc = self.eval(test_x, test_y, test_count)
        history.append(acc)
        print(history)
        pass

    def eval(self, data, label, test_count, show=True):
        # Calculate the probability and labels
        prob = self.forward(data)
        pred = self.classify(data)
        # Compute accuarcy
        correct_count = 0
        for itr_prob, itr_t, itr_pred in zip(prob, label, pred):
            # acc = 100. * itr_prob[0]
            # print("Pred: {:2.2f}% | GT: {:2d}".format(acc, itr_t))
            if itr_pred == itr_t:
                correct_count += 1
        # Percentage
        acc = 100. * correct_count / test_count
        #
        if show:
            print("Evaluation: ")
            print("Correct : {:3d} / {:3d}".format(correct_count, test_count))
            print("Accuracy: {:3.2f}%".format(acc))
        return acc
