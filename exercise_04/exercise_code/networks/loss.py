
import os
import pickle
import numpy as np
from exercise_code.networks.linear_model import *


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, y_out, y_truth):
        return NotImplementedError

    def backward(self, y_out, y_truth, upstream_grad=1.):
        return NotImplementedError

    def __call__(self, y_out, y_truth):
        loss = self.forward(y_out, y_truth)
        grad = self.backward(y_out, y_truth)
        return (loss, grad)


class L1(Loss):

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss for each sample of your training set.
        """
        result = None
        result = np.abs(y_out - y_truth)
        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of L1 loss gradients w.r.t y_out for
                  each sample of your training set.
        """
        gradient = None
        gradient = y_out - y_truth

        zero_loc = np.where(gradient == 0)
        negative_loc = np.where(gradient < 0)
        positive_loc = np.where(gradient > 0)

        gradient[zero_loc] = 0
        gradient[positive_loc] = 1
        gradient[negative_loc] = -1

        return gradient


class MSE(Loss):

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
                y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss for each sample of your training set.
        """
        result = None
        result = (y_out - y_truth)**2
        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of MSE loss gradients w.r.t y_out for
                  each sample of your training set.
        """
        gradient = None
        gradient = 2 * (y_out - y_truth)
        return gradient


class BCE(Loss):

    def forward(self, y_out, y_truth):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model.
                y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss for each sample of your training set.
        """
        result = None

        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass and return the output of the BCE loss.    #
        ########################################################################


        result = (-1 * y_truth) * np.log(y_out)-((1 - y_truth) * np.log(1 - y_out))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return result

    def backward(self, y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model.
               y_truth: [N, ] array ground truth value of your training set.
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out for
                  each sample of your training set.
        """
        gradient = None

        ########################################################################
        # TODO:                                                                #
        # Implement the backward pass. Return the gradient wrt y_out           #
        ########################################################################


        gradient = (-1*(y_truth / y_out) + ((1 - y_truth) / (1 - y_out)))

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return gradient
