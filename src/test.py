__author__ = 'victor'

import os
import sys
from scipy.signal import get_window
import pygame

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import TanhLayer

from plot_util import *

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

def play(fs, mX, pX):
    x = stft.stftSynth(mX, pX, M, H)
    play(fs, x)

def play(fs, x):
    pygame.init()
    outputfile = './output.wav'
    UF.wavwrite(x, fs, outputfile)
    if os.path.isfile(outputfile):
        sound = pygame.mixer.Sound(outputfile)
        sound.play()
    else:
        print "Output audio file not found", "The output audio file has not been computed yet"


soundfile1 = smstools_home + "/sounds/singing-female.wav"
soundfile2 = smstools_home + "/sounds/cello-double.wav"
M = 511 # window size
N = 512 # dft size (window + zero padding)
H = 200 # stft hop size
w = get_window("hamming", M)

def show(fs, X):
    print X.shape
    plot(fs, X, N, H)

def loadspec(soundfile, len=1):
    fs, x = UF.wavread(soundfile)
    x = resize(fs, x, len)
    w = get_window("hamming", M)
    mX, pX = stft.stftAnal(x, fs, w, N, H)
    # X time size ~ len x / hop size
    # X freq size ~ fft size / 2)
    return fs, x, mX, pX

# prepare dataset
partsize = 10
fs1, x1, mX1, pX1 = loadspec(soundfile1)
fs2, x2, mX2, pX2 = loadspec(soundfile2)
assert fs1 == fs2
xmix = np.add(0.5*x1, 0.5*x2)
mXmix, pXmix = stft.stftAnal(xmix, fs1, w, N, H)
# play(fs2, xmix)
# show(fs2, mX)

# learning
x1parts = split(mX1, partsize)
mixparts = split(mXmix, partsize)
assert len(x1parts) == len(mixparts)
netwidth = partsize * (N+2) / 2
net = buildNetwork(netwidth, 150, 150, netwidth, bias=True, hiddenclass=TanhLayer)
ds = SupervisedDataSet(netwidth, netwidth)
for i in np.arange(len(mixparts)):
    sample = np.reshape(mixparts[i], netwidth)
    target = np.reshape(x1parts[i], netwidth)
    ds.addSample(sample, target)
trainer = BackpropTrainer(net, ds)
plot_cont(trainer.train, 100)