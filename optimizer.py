import layers as Layers
import numpy as np


class Optimizer(object):

    def __init__(self, layers, lr=0.01):
        self.LR = lr
        self.layers = layers

    def backward(self, inp, grad):
        layers = list(reversed(self.layers))
        n_layers = np.shape(layers)[0]
        # print(n_layers)
        for i, layer in enumerate(layers):
            # print("INPUT FOR " + str(i) + " LAYER ", inp[n_layers - 1 - i])
            if isinstance(layer, Layers.FC):
                grad_w, grad_b, grad = layer.backward(inp[n_layers - 1 - i], grad)
                self.update(layer, grad_w, grad_b)
            elif isinstance(layer, Layers.ReLu):
                grad = layer.backward(inp[n_layers - 1 - i], grad)

    # def step(self, layers, outputs):
    #     outputs.reverse()
    #     layers.reverse()
    #     grad = Loss().get_loss(...)
    #
    #     for layer, output in zip(layers, outputs):
    #         grad = layer.backward(output, grad)
    #
    #     self.update()

    # @property
    def update(self, layer, grad_w, grad_b):
        # print("Layer Weight ", layer.weight.shape)
        # print("Weight gradient", grad_w.shape)
        #
        # print("Layer bias ", layer.bias.shape)
        # print("Bias gradient", grad_b.shape)

        # print("GRAD WEIGHT", grad_w)
        # print("GRAD BIAS", grad_b)

        layer.weight -= self.LR * grad_w
        layer.bias -= self.LR * grad_b




