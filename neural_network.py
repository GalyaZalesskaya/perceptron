import numpy as np
from copy import deepcopy
import layers


class NN(object):

    def __init__(self):
        self.layers = [layers.FC(512, 256), layers.ReLu(), layers.FC(256, 10)]
        self.outputs = []

    def forward(self, inp):
        self.outputs.append(deepcopy(inp))
        output = inp
        for layer in self.layers:
            output = layer.forward(output)
            self.outputs.append(deepcopy(output))
        return self.outputs

    # @property
    def get_layers(self):
        return self.layers




