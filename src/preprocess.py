from sklearn.decomposition import PCA
import numpy as np

from src.util import objwrite, objread


__author__ = 'victor'


# Approximate spectrogram magnitude statistics for all sounds in dataset
globalAvg = -80
globalScale = 3 * 21 # 3 * standard deviation
global_pca = objread('pca') # previously generated pca model fitting the training data


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

def fit_pca_model(flatStream, n, n_components=.999):
    x = flatStream.buffer(n)
    print 'running pca...'
    pca = PCA(n_components=n_components)
    pca.fit(x)

    orig_size = 2 * flatStream.flatWidth
    reduced_size = len(pca.components_)
    print 'components reduced from %d to %d' % (orig_size, reduced_size)
    return pca, orig_size, reduced_size

def write_pca_model(flatStream, n, n_components=.999):
    pca, orig_size, reduced_size = fit_pca_model(flatStream, n, n_components)
    name = 'pca_%d_to_%d' % (orig_size, reduced_size)
    objwrite(pca, name)
    return name

def pca_transform(x):
    return global_pca.transform(x)

def pca_inverse(xt):
    return global_pca.inverse_transform(xt)