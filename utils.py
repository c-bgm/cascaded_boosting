from skimage import io, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEBUG_FLAG = False


def savefig(samples, k, path):
    height, width = samples.shape[1], samples.shape[2]
    if len(samples.shape) == 3:
        ret = np.zeros((height * k, width * k), dtype=samples.dtype)
    elif len(samples.shape) == 4:
        channel = samples.shape[3]
        ret = np.zeros((height * k, width * k, channel), dtype=samples.dtype)
    else:
        print("[savefig][wrong samples shape]")
        exit(-1)
    for i in range(k):
        for j in range(k):
            n = i * k + j
            ret[i * height: (i + 1) * height, j * width: (j + 1) * width] = samples[n]
    ret = img_as_ubyte(ret)
    io.imsave(path, ret)


def debug(s):
    if DEBUG_FLAG:
        print(s)
    else:
        pass


def plot2d(arr):
    x = list(map(lambda a: a[0], arr))
    y = list(map(lambda a: a[1], arr))
    plt.scatter(x, y)
    plt.show()


def arr2img(arr, height, width, channel):
    shape = arr.shape
    if len(shape) == 4: # BCHW to BHWC
        if not shape[1] == channel:
            print("[arr2img] [channel %d]" % shape[1])
            exit(-1)
        else:
            arr = np.transpose(arr, (0, 2, 3, 1))
    img = arr.reshape(len(arr), -1)[:, :height*width*channel].reshape(-1, height, width, channel)
    if channel == 1:
        img = np.squeeze(img, -1)
    return img


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def outer(x, y):
    assert len(x.shape) == 2 and len(y.shape) == 2
    if isinstance(x, np.ndarray):
        res = []
        for xx, yy in zip(x, y):
            res.append(np.outer(xx, yy))
        return np.array(res)
    elif isinstance(x, torch.Tensor):
        return torch.bmm(x.unsqueeze(2), y.unsqueeze(1))


def dot_elementwise(x, y):
    assert len(x.shape) == 2 and len(y.shape) == 2
    if isinstance(x, np.ndarray):
        return np.sum(x*y, axis=1)
    elif isinstance(x, torch.Tensor):
        return torch.sum(x*y, dim=1)


def bernoulli_entropy(x):
    res = []
    for zz in x:
        temp = F.binary_cross_entropy(zz, zz, size_average=False)
        res.append(temp.unsqueeze(0))
    res = torch.cat(res, 0)
    return res


def bernoulli_ce(x, u):
    res = []
    for xx, uu in zip(x, u):
        temp = F.binary_cross_entropy(uu, xx, size_average=False)
        res.append(temp.unsqueeze(0))
    res = torch.cat(res, 0)
    return res


def display_elbo(elbos, elbo_thresholds):
    ELBOS = []
    ELBO_incs = []
    ELBOS.append(elbos[0])
    for i in range(len(elbo_thresholds)):
        elbo = ELBOS[-1]
        inc = elbos[i+1] - elbo_thresholds[i]
        ELBO_incs.append(inc)
        ELBOS.append(elbo + inc)
    print("ELBOS: {}".format(ELBOS))
    print("ELBO_incs: {}".format(ELBO_incs))


def RBM_config(X_dim, Z_dim, height, width, channel,
               epoch, max_epoch, sample_dir, model_dir, model_name,
               retrain, phase, batch_size=256, lr=0.1):
    return {'X_dim': X_dim, 'Z_dim': Z_dim,
            'height': height, 'width': width, 'channel': channel,
            'lr': lr, 'epoch': epoch, 'max_epoch': max_epoch, 'batch_size': batch_size,
            'sample_dir': sample_dir, 'model_dir': model_dir, 'model_name': model_name,
            'retrain': retrain, 'phase': phase}


def VAEBernoulli_config(X_dim, h_dim, Z_dim, height, width, channel,
                        sigma, epoch, max_epoch, sample_dir, model_dir, model_name,
                        retrain, phase, batch_size=256, lr=0.001):
    return {'X_dim': X_dim, 'h_dim': h_dim, 'Z_dim': Z_dim,
            'height': height, 'width': width, 'channel': channel, 'sigma': sigma,
            'lr': lr, 'epoch': epoch, 'max_epoch': max_epoch, 'batch_size': batch_size,
            'sample_dir': sample_dir, 'model_dir': model_dir, 'model_name': model_name,
            'retrain': retrain, 'phase': phase}


def VAEGauss_config(X_dim, h_dim, Z_dim, height, width, channel,
                    sigma, epoch, max_epoch, sample_dir, model_dir, model_name,
                    retrain, phase, batch_size=256, lr=0.001):
    return {'X_dim': X_dim, 'h_dim': h_dim, 'Z_dim': Z_dim,
            'height': height, 'width': width, 'channel': channel, 'sigma': sigma,
            'lr': lr, 'epoch': epoch, 'max_epoch': max_epoch, 'batch_size': batch_size,
            'sample_dir': sample_dir, 'model_dir': model_dir, 'model_name': model_name,
            'retrain': retrain, 'phase': phase}


def binarize(X):
    res = (X >= 0.5).astype(np.float32)
    return res


def dataset2Arr(dataset):
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)
    arr = []
    for batch in dataloader:
        arr.append(batch.detach().cpu().numpy())
    return np.concatenate(arr, axis=0)
