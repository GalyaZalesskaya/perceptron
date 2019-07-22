import os
import numpy as np
from PIL import Image


def download_data(path='/home/kotik/mnist_png', part="train"):
    # path += '/train'
    if part == 'train':
        path += '/training'
    else:
        path += '/testing'

    data = []
    data_labels = []
    for category in os.listdir(path):
        for image in os.listdir(path + '/' + category):
            data.append(np.array(Image.open(path + '/' + category + '/' + image)).reshape(-1))
            data_labels.append(int(category))

    data_labels = np.concatenate([data_labels])
    data = np.concatenate([data], axis=1)

    conc = list(zip(data, data_labels))
    np.random.seed(42)
    np.random.shuffle(conc)
    data, data_labels = zip(*conc)
    return np.array(data), np.array(data_labels)


class DataLoader(object):
    def __init__(self, batch_size, shuffle=False, path='/home/kotik/mnist_png', part='train'):
        self.batch_size = batch_size
        self.data, self.labels = download_data(path, part)
        self.n = self.labels.shape[0] // self.batch_size
        self.num = 0
        if shuffle:
            np.random.seed(None)
            conc = list(zip(self.data, self.data_labels))
            np.random.seed(42)
            np.random.shuffle(conc)
            self.data, self.data_labels = zip(*conc)
            self.data = np.array(self.data)
            self.data_labels = np.array(self.data_labels)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            start_index = self.batch_size * self.num
            end_index = start_index + self.batch_size
            self.num = self.num+1
            return self.data[start_index: end_index], self.labels[start_index: end_index]
        else:
            raise StopIteration()


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]