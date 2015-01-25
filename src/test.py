__author__ = 'victor'

import os
import sys
from scipy.signal import get_window

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import TanhLayer
import numpy as np

import util


smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft
import utilFunctions as UF


def split(mX, partsize):
    nparts = len(mX)/partsize
    Xparts = np.empty((nparts, partsize, mX.shape[1]))
    for i in np.arange(0, nparts):
        Xparts[i] = mX[i*partsize:(i+1)*partsize]
    return Xparts

def resize(fs, x, tsec):
    tsamples = tsec * fs
    if(len(x) >= tsamples):
        return x[:tsamples]
    else:
        rep = tsamples/len(x) + 1
        return np.tile(x, rep)[:tsamples]


soundfile1 = smstools_home + "/sounds/singing-female.wav"
soundfile2 = smstools_home + "/sounds/cello-double.wav"
N = 512 # dft size (window + zero padding)
M = N-1 # window size
H = (M+1)/2 # stft hop size
w = get_window("hamming", M)

def show(fs, X):
    print X.shape
    util.plot_stft(fs, X, N, H)

def writespec(fs, mX, pX, sync=False, outputfile='./output.wav', play=False):
    util.writespec(fs, mX, pX, M, H, sync, outputfile=outputfile, play=play)

def loadspec(soundfile, len=5):
    fs, x = UF.wavread(soundfile)
    x = resize(fs, x, len)
    w = get_window("hamming", M)
    mX, pX = stft.stftAnal(x, fs, w, N, H)
    # X time size ~ len x / hop size
    # X freq size ~ fft size / 2)
    return fs, x, mX, pX


# prepare dataset
fs1, x1, mX1, pX1 = loadspec(soundfile1)
fs2, x2, mX2, pX2 = loadspec(soundfile2)
assert fs1 == fs2
xmix = np.add(0.5*x1, 0.5*x2)
mXmix, pXmix = stft.stftAnal(xmix, fs1, w, N, H)
# util.play(fs2, xmix)
writespec(fs2, mXmix, pXmix, outputfile='./mix.wav')
writespec(fs2, mX1, pX1, outputfile='./target.wav')
# show(fs2, mXmix)
# play_spec(fs2, mXmix, np.zeros(pXmix.shape))
partlength = 50
mX1parts = split(mX1, partlength)
pX1parts = split(pX1, partlength)
mXmixparts = split(mXmix, partlength)
pXmixparts = split(pXmix, partlength)
assert len(mX1parts) == len(mXmixparts) == len(pX1parts) == len(pXmixparts)
nparts = len(mX1parts)
freqrange = (N+2) / 2 # idk why this value exactly. found by try and error


# train
epochs = 30
flatwidth = partlength * freqrange
netwidth = 2 * flatwidth
def flatten_sample(v1, v2):
    res1 = np.reshape(v1, flatwidth)
    res2 = np.reshape(v2, flatwidth)
    return np.append(res1, res2)
net = buildNetwork(netwidth, 150, netwidth, bias=True, hiddenclass=TanhLayer)
ds = SupervisedDataSet(netwidth, netwidth)
for i in np.arange(nparts):
    sample = flatten_sample(mXmixparts[i], pXmixparts[i])
    target = flatten_sample(mX1parts[i], pX1parts[i])
    ds.addSample(sample, target)
trainer = BackpropTrainer(net, ds)
util.plot_cont(trainer.train, epochs)


# filter
def unflatten_sample(v):
    flatm = v[:flatwidth]
    flatp = v[flatwidth:]
    mX = np.reshape(flatm, (partlength, freqrange))
    pX = np.reshape(flatp, (partlength, freqrange))
    return mX, pX
mXresult = np.empty((partlength, freqrange))
pXresult = np.empty((partlength, freqrange))
for i in np.arange(nparts):
    sample = flatten_sample(mXmixparts[i], pXmixparts[i])
    netout = net.activate(sample)
    mXpart, pXpart = unflatten_sample(netout)
    mXresult = np.append(mXresult, mXpart, axis=0)
    pXresult = np.append(pXresult, pXpart, axis=0)


# show result
gap = len(mXresult) - len(mX1)
mXresult = mXresult[0:len(mXresult)-gap]
pXresult = pXresult[0:len(pXresult)-gap]
writespec(fs2, mXresult, pXresult, sync=True, outputfile='./output.wav')
