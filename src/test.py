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
x1parts = split(mX1, partlength)
mixparts = split(mXmix, partlength)
assert len(x1parts) == len(mixparts)
nparts = len(x1parts)
freqrange = (N+2) / 2 # idk why this value exactly. found by try and error


# train
epochs = 30
netwidth = partlength * freqrange
net = buildNetwork(netwidth, 150, netwidth, bias=True, hiddenclass=TanhLayer)
ds = SupervisedDataSet(netwidth, netwidth)
for i in np.arange(nparts):
    sample = np.reshape(mixparts[i], netwidth)
    target = np.reshape(x1parts[i], netwidth)
    ds.addSample(sample, target)
trainer = BackpropTrainer(net, ds)
util.plot_cont(trainer.train, epochs)

# filter
result = np.empty((partlength, freqrange))
for i in np.arange(nparts):
    sample = np.reshape(mixparts[i], netwidth)
    netout = net.activate(sample)
    part = np.reshape(netout, (partlength, freqrange))
    result = np.append(result, part, axis=0)

# play result
gap = len(result) - len(pX1)
result = result[0:len(result)-gap]
writespec(fs2, result, pX1, sync=True, outputfile='./output.wav')
