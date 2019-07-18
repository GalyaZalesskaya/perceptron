from abc import ABC, abstractmethod
import numpy as np


class BaseLayer(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def forward(self, inp):
        pass

    @abstractmethod
    def backward(self, inp, grad):
        pass


class FC(BaseLayer):

    def __init__(self, inp_size, out_size):
        self.weight = np.random.normal(loc=0.0, scale=np.sqrt(2/(inp_size+out_size)), size=(inp_size, out_size))
        self.bias = np.zeros(out_size)

    def forward(self, inp):
        return np.dot(inp, self.weight) + self.bias

    def backward(self, inp, grad):
        # grad_w = np.dot(grad.T, inp)  #first variant
        grad_w = np.dot(inp.T, grad)
        grad_b = np.mean(grad, axis=0) * inp.shape[0]
        grad_inp = np.dot(grad, self.weight.T)
        return grad_w, grad_b, grad_inp


class ReLu(BaseLayer):

    def __init__(self):
        pass

    def forward(self, inp):
        return np.maximum(0, inp)

    def backward(self, inp, grad):
        return grad * (inp > 0)


# fc = FC(10, 20)
# inp = np.arange(40).reshape(2, -1)
# grad = np.arange(40).reshape(2, -1) - 4
# print(fc.backward(inp, grad))
# #
# relu = ReLu()
# inp -= 5
# print(relu.backward(inp, grad))
