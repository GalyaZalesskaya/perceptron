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
        return np.sum((y-softmax(output))**2, axis=1)  # returns vector of losses for every picture
        # return np.mean(np.sum((y - softmax(output)) ** 2, axis=1))  # returns one loss for batch

    def get_grad_loss(self, labels, output, batch_size):
        y = one_hot(labels)
        # print("GRAD LOSS ", np.sum((y - output), axis=1)*2/batch_size)
        return np.sum((y - output), axis=1)*2/batch_size

    def get_grad(self, labels, output):
        batch_size = labels.shape[0]
        feature = output.shape[1]
        JS = []
        output = softmax(output)
        for picture in output:
            # picture = softmax(picture)
            Jx = -picture*picture.T * (-np.eye(feature) + 1) + np.eye(feature)*(picture - picture**2)
            # print("PICTURE ", picture, Jx)
            JS.append(Jx)
        JS = np.stack(JS)  # m*n*n

        grad_loss = self.get_grad_loss(labels, output, batch_size)  # m*1

        print("GRAD LOSS ", grad_loss)
        JS = JS * grad_loss[:, np.newaxis, np.newaxis]  # m*n*n

        return np.mean(JS, axis=-1)  # m*n


def softmax(x, ax=1):
    return np.exp(x)/np.sum(np.exp(x), axis=ax).reshape(-1, 1)


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

