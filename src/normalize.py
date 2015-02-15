__author__ = 'victor'

import numpy as np

# Approximate spectrogram magnitude statistics for all sounds in dataset
globalAvg = -80
globalScale = 3 * 21 # 3 * standard deviation


def normalize_gauss(mX):
    avg = np.average(mX)
    std = np.std(mX)
    mXnorm = (mX - avg) / std
    return mXnorm, avg, std

def normalize_linear(mX):
    avg = np.average(mX)
    range = (np.max(mX) - np.min(mX)) / 2
    mXnorm = (mX - avg) / range
    return mXnorm, avg, range

def normalize_static(mX):
    '''
    Approximate spectrogram magnitude mean and std for all sounds in dataset
    '''
    return (mX - globalAvg) / globalScale

def unnormalize(mXnorm, avg, std):
    mX = mXnorm * std + avg
    return mX

def unnormalize_static(mXnorm):
    return unnormalize(mXnorm, globalAvg, globalScale)


class OnlineNorm:
    '''
    Computes mean and std online
    '''

    def __init__(self):
        self.n = 0
        self.mean = 0
        self.m2 = 0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    def stat(self):
        if self.n < 2:
            return self.mean, 0

        variance = self.m2/(self.n - 1)
        std = np.sqrt(variance)
        return self.mean, std

    def norm(self, x):
        mean, std = self.stat()
        return (x - mean) / std

    def unorm(self, x):
        mean, std = self.stat()
        return x * std + mean

    def push(self, x):
        self.update(x)
        return self.norm(x)


class BatchNorm(OnlineNorm):
    '''
    Computes mean and std in batches, passing the batch elements to OnlineNorm one by one
    '''

    def __init__(self):
        OnlineNorm.__init__(self)

    def batchUpdate(self, x):
        flatX = np.reshape(x, -1)
        for xi in flatX:
            self.update(xi)

    def batchPush(self, x):
        self.batchUpdate(x)
        return self.norm(x)
