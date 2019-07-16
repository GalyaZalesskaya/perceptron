import os
import numpy as np
from PIL import Image


def download_data(path='/home/kotik/mnist_png', part="train"):
    if part == 'train':
        path += '/training'
    else:
        path += '/testing'

    data = []
    data_labels = []
    for category in os.listdir(path):
        for image in os.listdir(path + '/' + category):
            data.append(np.array(Image.open(path + '/' + category + '/' + image)).reshape(-1))
            data_labels.append(category)

    data_labels = np.concatenate([data_labels])
    data = np.concatenate([data], axis=1)
    return data, data_labels


class DataLoader(object):
    def __init__(self, batch_size, shuffle=False, path='/home/kotik/mnist_png', part='train'):
        self.batch_size = batch_size
        self.data, self.labels = download_data(path, part)
        self.n = self.labels.shape[0] // self.batch_size
        self.num = 0

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


