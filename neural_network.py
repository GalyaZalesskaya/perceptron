import numpy as np
from copy import deepcopy
import layers
import data_loader
import loss
import optimizer
from keras.datasets import mnist


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

    # @property
    def get_layers(self):
        return self.layers


def validate(data_loader, nn):
    accuracy = []  # vector with accuracies for every batch
    i = 0
    for batch in data_loader:
        prediction = nn.forward(batch[0])[-1]
        # print(prediction)
        # print(prediction.shape)
        prediction = prediction.argmax(axis=1)
        # print(list(zip(prediction, batch[-1])))
        print("PREDICTON ", prediction)
        print("LABEL ", batch[-1])
        print(prediction == batch[-1])
        accuracy.append(np.mean(prediction == batch[-1]))
        # print(accuracy)

        i += 1
        if i == 5:
            break
    return np.mean(accuracy)


def train(data_loader, lr, nn):
    # batch_size = data_loader.batch_size
    i = 0
    accuracy = []
    for x, y in data_loader:
        batch_loss = []

        # print("FC1 weights ", nn.get_layers()[0].weight)
        #
        # print("FC2 weights ", nn.get_layers()[-1].weight)
        # print("_______________________________________")

        output = nn.forward(x)
        # print("OUTPUTS OF FORWARD ", output)
        prediction = output[-1]
        l2 = loss.L2()

        # print("batch shape ", batch[0], np.shape(batch[-1]))
        # print("\nOutput shape", output,  np.shape(output))
        # print("batch shape \nOutput shape", np.shape(batch[-1]), np.shape(output))
        loss_for_batch = l2.get_loss(y, prediction)
        # print(str(i) + " PREDICTON ", prediction.argmax(axis=1))
        # print(str(i) + " LABEL ", batch[-1])
        # print(prediction.argmax(axis=1) == batch[-1])
        accuracy.append(np.mean(prediction.argmax(axis=1) == y))

        # print("LOSS FOR " + str(i) + " BATCH ", loss_for_batch)
        batch_loss.append(np.mean(loss_for_batch))
        # print(np.mean(batch_loss))

        grad = l2.get_grad(y, prediction)
        # print("Gradient of loss ", grad, grad.shape)
        opt = optimizer.Optimizer(nn.get_layers(), lr)
        # print()
        opt.backward(output, grad)


        # print("FC1 weights ", nn.get_layers()[0].weight)
        #
        # print("FC2 weights ", nn.get_layers()[-1].weight)
        # i += 1
        # if i == 5:
        #     break
    print("Loss is ", np.mean(batch_loss))
    print("Accuracy is ", np.mean(accuracy))




(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
# loader = data_loader.DataLoader(16, part='test')

nn = NN()
for e in range(50):
    # loader = data_loader.DataLoader(16, part='test')
    # train(loader, 0.01, nn)
    print("Epoch ", str(e))
    train(data_loader.iterate_minibatches(x_train, y_train, batchsize=64, shuffle=False), 0.01, nn)

    # print("ACCURACY IS ", validate(loader, nn))

# def train_model(data_loader, nn, batch_size, lr, epochs):
#     for epoch in range(epochs):
#         train(data_loader, lr, nn)
#
# def main():
#     lr = 0.01
#     batch_size = 32
#     epochs = 2
#

