from sklearn.decomposition import PCA
import numpy as np

from src.util import objwrite


__author__ = 'victor'


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

def unnormalize(x, avg, std):
    return x * std + avg

def pca_fit(flatStream, n, n_components=.999):
    x = flatStream.buffer(n)
    print 'running pca...'
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(x)

    orig_size = flatStream.width
    reduced_size = len(pca.components_)
    print 'components reduced from %d to %d' % (orig_size, reduced_size)
    return pca, orig_size, reduced_size

def pca_fit_write(flatStream, n, n_components=.999):
    pca, orig_size, reduced_size = pca_fit(flatStream, n, n_components)
    name = 'pca_%d_to_%d' % (orig_size, reduced_size)
    objwrite(pca, name)
    return name
