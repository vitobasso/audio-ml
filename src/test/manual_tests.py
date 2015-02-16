import os
import sys

import numpy as np

from settings import SMSTOOLS_MODELS, SAMPLES_HOME
from src.datasource import MixedSpectrumStream, FlatStream, MPStandardStream, PcaStream, StandardStream
from src.fourrier import Fourrier
from src.preprocess import pca_fit_write, scaler_fit_write
from src.util import play, objread


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
        mXi, pXi = flatStream._unflatten(flat_x[i])
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
# normStream = NormSpecStream(mixSpec)
# flatStream = FlatStream(normStream)
# pca = objread('pca')
# pcaStream = PcaStream(flatStream, pca)



# flat_x = flatStream.buffer(1000, 0)
# mX, pX = flatStream.unflattenMany(flat_x)
# flatStream.spectStream.fourrier.write(mX, pX)
# play(sync=True)

# pca_fit_write(flatStream, 1000)


def test_streams():
    # create streams
    timeWidth = 1 # num of spectrogram time steps to input to the net each time
    fourrier = Fourrier()
    specMix = MixedSpectrumStream('guitar', 'drums', timeWidth, fourrier)
    specTar = specMix.subStream1()
    scaler = objread('scaler_514')
    flatMix = FlatStream(specMix)
    flatTar = FlatStream(specTar)
    normMix = StandardStream(flatMix, scaler)
    normTar = StandardStream(flatTar, scaler)
    pca = objread('pca_514_to_452_w')
    pcaMix = PcaStream(normMix, pca)
    pcaTar = PcaStream(normTar, pca)

    # do and undo
    x = pcaMix.buffer(2000, 10000)
    x = pcaMix.restore(x)
    x = normMix.unstandardMany(x)
    mX, pX = flatMix.unflattenMany(x)
    fourrier.plot(mX)
    fourrier.write(mX, pX)
    play(sync=True)

def create_pca():
    # create streams
    timeWidth = 1 # num of spectrogram time steps to input to the net each time
    fourrier = Fourrier()
    specMix = MixedSpectrumStream('piano', 'acapella', timeWidth, fourrier)
    specTar = specMix.subStream1()
    normMix = MPStandardStream(specMix)
    normTar = MPStandardStream(specTar)
    flatMix = FlatStream(normMix)
    flatTar = FlatStream(normTar)

    # fit pca
    pca_fit_write(flatMix, 5000, whiten=True)

def create_scaler():
    # create streams
    timeWidth = 1 # num of spectrogram time steps to input to the net each time
    fourrier = Fourrier()
    specMix = MixedSpectrumStream('piano', 'acapella', timeWidth, fourrier)
    specTar = specMix.subStream1()
    flatMix = FlatStream(specMix)
    flatTar = FlatStream(specTar)

    # fit scaler
    scaler_fit_write(flatMix, 5000)

# create_scaler()
test_streams()