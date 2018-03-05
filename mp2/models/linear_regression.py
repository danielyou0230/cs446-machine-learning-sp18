"""Implements linear regression."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class LinearRegression(LinearModel):
    """Implements a linear regression mode model."""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        with respect to w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).

        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,1).
        """
        # grad = x.T * (f - y) - lambda * w
        # Add bias (1) to x
        _x = np.concatenate((self.x, np.ones((self.x.shape[0], 1))), axis=1)
        # print ("_x       shape: {0}".format(_x.shape))
        # print (" w       shape: {0}".format(self.w.shape))
        # Calculate gradient
        total_grad = _x.T.dot(f - y) + self.w_decay_factor * self.w
        # print ("gradient shape: {0}".format(total_grad.shape))
        # print ("weights  shape: {0}".format(self.w.shape))
        return total_grad

    def total_loss(self, f, y):
        """Computes the total loss, square loss + L2 regularization.

        Overall loss is sum of squared_loss + w_decay_factor*l2_loss
        Note: Don't forget the 0.5 in the squared_loss!

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum square loss + reguarlization.
        """
        squared_loss = np.sum(np.square(f - y))
        w_norm = self.w.T.dot(self.w)[0][0]
        # print ("squared_loss {0}".format(np.sum(squared_loss)))
        total_loss = 0.5 * squared_loss + self.w_decay_factor * 0.5 * w_norm
        # print ("loss {0}".format(total_loss))
        return total_loss

    def predict(self, f):
        """Nothing to do here.
        """
        return f
