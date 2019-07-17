import layers as Layers
import numpy as np


class Optimizer(object):

    def __init__(self, layers, lr=0.01):
        self.LR = lr
        self.layers = layers

    def backward(self, inp, grad):
        layers = list(reversed(self.layers))
        for layer in layers:
            if isinstance(layer, Layers.FC):
                grad_w, grad_b, grad = layer.baskward(layer, inp, grad)
                self.update(inp, grad_w, grad_b)
            elif isinstance(layer, Layers.ReLu):
                grad = layer.backward(inp, grad)

    # def step(self, layers, outputs):
    #     outputs.reverse()
    #     layers.reverse()
    #     grad = Loss().get_loss(...)
    #
    #     for layer, output in zip(layers, outputs):
    #         grad = layer.backward(output, grad)
    #
    #     self.update()

    @property
    def update(self, layer, grad_w, grad_b):
        layer.weight -= self.LR * grad_w
        layer.bias -= self.LR * grad_b




