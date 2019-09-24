from torch.utils.data import Dataset, DataLoader
from skimage import io
import os
import numpy as np


def load_mnist_2d(data_dir):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX / 255.0
    teX = teX / 255.0

    return trX, teX, trY, teY


def load_mnist_4d(data_dir):
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 1, 28, 28)).astype(np.float32)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 1, 28, 28)).astype(np.float32)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX / 255.0
    teX = teX / 255.0

    return trX, teX, trY, teY


class QuickDataset(Dataset):
    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, item):
        return self.arr[item]


class LabelledDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]


class CelebaDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 202599

    def __getitem__(self, item):
        img = io.imread(os.path.join(self.path, "%06d.jpg" % (item + 1))) / 255.
        img = np.transpose(img, (2, 0, 1))
        return img.astype(np.float32)


class CelebaFlatDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return 202599

    def __getitem__(self, item):
        img = io.imread(os.path.join(self.path, "%06d.jpg" % (item + 1))) / 255.
        img = img.astype(np.float32)
        return img.reshape(-1)


class GaussDataset(Dataset):
    def __init__(self, mu, var, dim):
        self.mu = mu
        self.var = var
        self.dim = dim
        pass

    def __len__(self):
        return 1000

    def __getitem__(self, item):
        eps = np.random.normal(loc=0., scale=1., size=[self.dim])
        return self.mu + eps * np.sqrt(self.var)


if __name__ == "__main__":
    pass

