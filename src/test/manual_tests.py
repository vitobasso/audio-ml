import os
import sys

import numpy as np

from settings import SMSTOOLS_MODELS, SAMPLES_HOME
from src.fourrier import Fourrier
from src.preprocess import unnormalize_static


sys.path.append(SMSTOOLS_MODELS)
import utilFunctions as uf

__author__ = 'victor'


def wavstat(foldername):
    '''
    Compute statistics about the audio files
    '''
    folderpath = SAMPLES_HOME + foldername
    result = np.array([])
    fourrier = Fourrier()
    for path, dirs, files in os.walk(folderpath):
        for f in files:
            fpath = folderpath + '/' + f
            fs, x = uf.wavread(fpath)
            mX, pX = fourrier.analysis(x)
            result = np.append(result, pX)
    print np.average(result), np.std(result), np.min(result), np.max(result)

#                   mX              pX              x
#                   mean    std     min     max     mean    std
# guitar:           -72     23      -276    477     -3e-05  0.02
# drums:            -70     22      -593    320     8e-6    0.2
# piano:            -94     19                      3e-6    0.08
# acapella:         -83     18                      -6e-4   0.1
# violin:           -87     23                      -8e-05  0.09


def flatspecwrite(flatStream, n, flat_x):
    print 'unfolding and saving...'
    timeWidth, freqRange = flatStream.spectStream.shape
    shape = (n * timeWidth, freqRange)
    mX = np.array([])
    pX = np.array([])
    for i in range(n):
        mXi, pXi = flatStream.unflatten(flat_x[i])
        mX = np.append(mX, mXi)
        pX = np.append(pX, pXi)
    mX = np.reshape(mX, shape)
    pX = np.reshape(pX, shape)
    assert mX.shape == pX.shape == shape
    mX = unnormalize_static(mX)
    print 'writing...'
    flatStream.spectStream.fourrier.write(mX, pX)


def pca_test(pca, x, flatStream):
    n = len(x)
    xt = pca.transform(x)
    xr = pca.inverse_transform(xt)
    flatspecwrite(flatStream, n, xr)


# mixSpec = MixedSpectrumStream('piano', 'acapella', 1)
# mixSpec = MixedSpectrumStream('drums', 'guitar', 1)
# spec = mixSpec.subStream1()
# flatStream = FlatStream(mixSpec)

# flat_x = flatStream.buffer(1000)
# mX, pX = flatStream.unflattenMany(flat_x)
# flatStream.spectStream.fourrier.write(mX, pX)
# play(sync=True)