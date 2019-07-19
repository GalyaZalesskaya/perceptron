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
        # print("Labels shape, one-hot shape ", labels.shape, y.shape)
        # print("REAL ", y)
        # print("PREDICTED ", softmax(output))
        return np.sum((y-softmax(output))**2, axis=1)  # returns vector of losses for every picture
        # return np.mean(np.sum((y - softmax(output)) ** 2, axis=1))  # returns one loss for batch

    def get_grad_loss(self, labels, output, batch_size):
        y = one_hot(labels)
        # print("GRAD LOSS ", np.sum((y - output), axis=1)*2/batch_size)
        return (y - output)*2/batch_size

    def get_grad(self, labels, output):
        batch_size = labels.shape[0]
        feature = output.shape[1]
        JS = []
        output = softmax(output)
        for picture in output:
            # picture = softmax(picture)
            Jx = -picture*picture.T * (-np.eye(feature) + 1) + np.eye(feature)*(picture - picture**2)
            # print("PICTURE ", picture, Jx)
            JS.append(np.sum(Jx))  # mean number
        JS = np.stack(JS)  # 1*m

        grad_loss = self.get_grad_loss(labels, output, batch_size)  # m*n

        # print("GRAD LOSS ", grad_loss)

        JS = JS[:, np.newaxis] * grad_loss  # m*1 * m*n = m*n(broadcasting)

        return JS  # m*n


def softmax(x, ax=1):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / e_x.sum(axis=1).reshape(-1, 1)
    # return np.exp(x)/np.sum(np.exp(x), axis=ax).reshape(-1, 1)


def one_hot(labels):
    n_values = max(np.max(labels) + 1, 10)
    return np.eye(n_values)[labels]


# out = np.arange(0, 20).reshape(4, 5)
# lab = np.array([4, 4, 1, 4])
# l2 = L2()
# print(l2.get_loss(lab, out))
# print(softmax(out))
# print(l2.get_grad_loss(lab, out))
# print(l2.get_grad(lab, out))

