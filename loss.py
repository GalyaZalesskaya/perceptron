from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_loss(self, labels, output):
        pass

    @abstractmethod
    def get_grad(self, labels, output):
        pass


class L2(Loss):

    def get_loss(self, labels, output):
        y = one_hot(labels)
        return (y-softmax(output))**2 / labels.shape[0]

    def get_grad(self, labels, output):         # PROBLEMS HERE - DISCUSS
        # batch_size = labels.shape[0]
        # grad_softmax = -output*output.T * (-np.eye(batch_size) + 1) + np.eye(batch_size)*(output - output**2)
        # y = one_hot(labels)
        # return -2/batch_size( y - soft)

        return 1


def softmax(x, ax=1):
    return np.exp(x)/np.sum(np.exp(x), axis=ax).reshape(-1, 1)


def one_hot(labels):
    n_values = np.max(labels) + 1
    return np.eye(n_values)[labels]
