import numpy as np
from copy import deepcopy
import layers
import data_loader


class NN(object):

    def __init__(self):
        self.layers = [layers.FC(28*28, 2048), layers.ReLu(), layers.FC(2048, 10)]
        self.outputs = []

    def forward(self, inp):
        self.outputs.append(deepcopy(inp))
        output = inp
        for layer in self.layers:
            output = layer.forward(output)
            self.outputs.append(deepcopy(output))
        return self.outputs

    @property
    def get_layers(self):
        return self.layers


def validate(data_loader, nn):
    accuracy = []  # vector with accuracies for every batch
    for batch in data_loader:
        prediction = nn.forward(batch[0])[-1]
        # print(prediction)
        # print(prediction.shape)
        prediction = prediction.argmax(axis=1)
        # print(list(zip(prediction, batch[-1])))
        accuracy.append(np.mean(prediction == batch[-1]))
        # print(accuracy)

    return np.mean(accuracy)

# def train(data_loader, lr, nn):

    # for batch in data_loader:
        ###  CONTINUE TOMORROW


# loader = data_loader.DataLoader(256, part='test')
# nn = NN()
#
# print(validate(loader, nn))


