"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        # Append bias to the data first
        bias = np.ones((self.x.shape[0], 1))
        _x = np.concatenate((self.x, bias), axis=1)
        # regularization gradient = (1 - lambda) * w
        # reg_grad = (1 - self.w_decay_factor) * self.w
        reg_grad = self.w_decay_factor * self.w
        #
        indicator = np.array([1 if itr < 1 else 0 for itr in y * f])
        loss_grad = _x.T.dot(y * indicator[:, np.newaxis])
        total_grad = reg_grad - loss_grad
        # print("reg_grad: {0}, loss_grad: {1}".format(reg_grad, loss_grad))
        return total_grad

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor*l2_loss

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        def Hinge_Loss(a):
            """
            Calculate hinge loss
            Args:
                a(numpy.ndarray): input array for calculating hinge loss
            Returns:
                loss(float): accumulative hinge loss with given input
            """
            loss = sum([max(0, 1 - itr[0]) for itr in a])
            return loss
        #
        hinge_loss = Hinge_Loss(f)
        l2_loss = 0.5 * self.w_decay_factor * np.linalg.norm(self.w) ** 2
        #
        total_loss = hinge_loss + l2_loss
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,). Tie break 0 to 1.0.
        """
        y_predict = np.array([+1 if itr > 0 else -1 for itr in f])
        return y_predict
