import layers as Layers
import numpy as np

# AGAIN, I DONT UNDERSTAND


class Optimizer(object):

    def __init__(self, layers):
        self.LR = 0.01
        self.grads = [] # SAVE GRADIENTS?????
        for layer in layers:
            if isinstance(layer, Layers.FC):
                pass
            if isinstance(layer, Layers.ReLu):
                pass

    def step(self, layers, outputs):
        outputs.reverse()
        layers.reverse()
        grad = Loss().get_loss(...)

        for layer, output in zip(layers, outputs):
            grad = layer.backward(output, grad)

        self.update()

    @property
    def update(self):
        # grad = 0.9 * prev_grad + 0.1 * grad
        # LR???
        pass



