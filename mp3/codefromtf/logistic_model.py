"""logistic model class for binary classification."""
import tensorflow as tf
import numpy as np
from math import ceil


class LogisticModel_TF(object):
    def __init__(self, ndims, W_init='zeros'):
        """Initialize a logistic model.

        This function prepares an initialized logistic model.
        It will initialize the weight vector, self.W, based on the method
        specified in W_init.

        We assume that the FIRST index of Weight is the bias term,
            Weight = [Bias, W1, W2, W3, ...]
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
        self.ndims = ndims
        self.W_init = W_init
        self.W0 = None
        ###############################################################
        # Fill your code below
        ###############################################################
        w_shape = [self.ndims + 1, 1]
        if W_init == 'zeros':
            # Hint: self.W0 = tf.zeros([self.ndims+1, 1])
            self.W0 = tf.zeros(shape=w_shape)
            pass
        elif W_init == 'ones':
            self.W0 = tf.ones(shape=w_shape)
            pass
        elif W_init == 'uniform':
            self.W0 = tf.random_uniform(shape=w_shape, minval=0.0, maxval=1.0)
            pass
        elif W_init == 'gaussian':
            self.W0 = tf.random_normal(shape=w_shape, mean=0., stddev=0.1)
            pass
        else:
            print('Unknown W_init ', W_init)

    def build_graph(self, learn_rate):
        """ build tensorflow training graph for logistic model.
        Args:
            learn_rate: learn rate for gradient descent
            ......: append as many arguments as you want
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        # Hint: self.W = tf.Variable(self.W0)
        self.W = tf.Variable(self.W0)
        self.inputs = tf.placeholder(tf.float32, shape=[None, self.ndims + 1])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        self.pred = tf.sigmoid(tf.matmul(self.inputs, self.W))
        # Evaluate model
        self.correct = tf.equal(tf.round(self.pred), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
        #
        self.cost = tf.losses.mean_squared_error(self.labels, self.pred)
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate)
        self.init_graph = tf.global_variables_initializer()
        pass

    def fit(self, Y_true, X, max_iters, batch_size=50):
        """ train model with input dataset using gradient descent.
        Args:
            Y_true(numpy.ndarray): dataset labels with a dimension of
                                (# of samples, 1)
            X(numpy.ndarray): input dataset with a dimension of
                                (# of samples, ndims + 1)
            max_iters: maximal number of training iterations
            ......: append as many arguments as you want
        Returns:
            (numpy.ndarray): sigmoid output from well trained logistic model,
                             used for classification
                             with a dimension of (# of samples, 1)
        """
        ###############################################################
        # Fill your code in this function
        ###############################################################
        test_count = 200
        history = list()
        #
        print("Training Info:")
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
        # collect some operations
        opt_operation = [self.cost, self.optimizer.minimize(self.cost)]
        eval_operation = [self.accuracy, self.pred]
        #
        with tf.Session() as sess:
            sess.run(self.init_graph)
            for step in range(max_iters):
                print("Step: {:5d}/{:5d}".format(step + 1, max_iters), end='')
                # shuffle the training data at each epoch
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
                # calculate loss
                loss, _ = sess.run(opt_operation,
                                   feed_dict={self.inputs: batch_x,
                                              self.labels: batch_y})
                print(" | Loss: {:5.7f}".format(loss), end='')
                # calculate accuracy
                acc, pred = sess.run(eval_operation,
                                     feed_dict={self.inputs: batch_x,
                                                self.labels: batch_y})
                print(" | Training Acc: {:3.2f}%".format(acc * 100.))
                # test each 500 steps
                if step > 0 and step % 500 == 0:
                    acc, pred = sess.run(eval_operation,
                                         feed_dict={self.inputs: test_x,
                                                    self.labels: test_y})
                    history.append(acc)
            # final test
            acc, _ = sess.run(eval_operation,
                              feed_dict={self.inputs: test_x,
                                         self.labels: test_y})
            history.append(acc)
            print(history)
            # total accuracy
            acc, score = sess.run(eval_operation,
                                  feed_dict={self.inputs: X,
                                             self.labels: Y_true})
        #
        return score
