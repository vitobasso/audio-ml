__author__ = 'victor'
import pygame
import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from pybrain.tools.xml import NetworkWriter, NetworkReader


smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft
import utilFunctions as uf

output_dir = '../out/'
fs = 44100


def plot_wav(x):
    plt.plot(x)
    plt.show()


'''
plot spectrogram
'''
def plot_stft(mX, N, H):
    # create figure to plot
    plt.figure(figsize=(6, 3))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot magnitude spectrogram
    plt.subplot(1, 1, 1)
    numFrames = int(mX[:, 0].size)
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = fs * np.arange(N * maxplotfreq / fs) / N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :N * maxplotfreq / fs + 1]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('magnitude spectrogram')
    plt.autoscale(tight=True)

    plt.tight_layout()
    plt.show()


'''
plot signal and spectrogram
'''
def plot_wav_stft(x, mX, N, H):
    # create figure to plot
    plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot the input sound
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(x.size) / float(fs), x)
    plt.axis([0, x.size / float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot magnitude spectrogram
    plt.subplot(2, 1, 2)
    numFrames = int(mX[:, 0].size)
    frmTime = H * np.arange(numFrames) / float(fs)
    binFreq = fs * np.arange(N * maxplotfreq / fs) / N
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:, :N * maxplotfreq / fs + 1]))
    plt.xlabel('time (sec)')
    plt.ylabel('frequency (Hz)')
    plt.title('magnitude spectrogram')
    plt.autoscale(tight=True)

    plt.tight_layout()
    plt.show()


'''
plot 1d-function updating continuously
fun: function
xmax: maximum value of X, when updates must stop
'''
def plot_cont(fun, xmax):
    y = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def update(i):
        t1 = time.time()
        yi = fun()
        dt = time.time() - t1

        y.append(yi)
        x = range(len(y))
        ax.clear()
        ax.plot(x, y)

        print '%d: err=%f, t=%.2fs' % (i, yi, dt)

    a = anim.FuncAnimation(fig, update, frames=xmax, repeat=False)
    # reference 'a' above is needed to avoid garbage collector from removing the obj
    plt.show()


def wavwrite(x, outputfile='output.wav'):
    file = output_dir + outputfile
    uf.wavwrite(x, fs, file)


'''
write sound file (given as spectrogram)
fs: frame rate (sec)
mX: magnitude spectrum
pX: phase spectrum
M: window size
H: hop size
'''
def specwrite(mX, pX, M, H, outputfile='output.wav'):
    file = output_dir + outputfile
    x = stft.stftSynth(mX, pX, M, H)
    uf.wavwrite(x, fs, file)


def play(soundfile='output.wav', sync=False):
    file = output_dir + soundfile
    pygame.init()
    if os.path.isfile(file):
        if(play):
            sound = pygame.mixer.Sound(file)
            ch = sound.play()
            if(sync):
                wait_sound(ch)
    else:
        print "Output audio file not found", "The output audio file has not been computed yet"


def wait_sound(ch):
    while ch.get_busy():
        time.sleep(1)


def savenet(net, width, err):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H:%M:%S')
    file = '%snet_%d_%f_%s' % (output_dir, width, err, timestamp)
    NetworkWriter.writeToFile(net, file)


def loadnet(filename):
    file = output_dir + filename
    print 'loading net: ' + filename
    return NetworkReader.readFrom(file)


def split_spec_sub(matrix, length):
    n = len(matrix)/length
    parts = np.empty((n, length, matrix.shape[1]))
    for i in np.arange(0, n):
        parts[i] = matrix[i*length:(i+1)*length]
    return parts

def split_spec(mX, pX, length):
    mXparts = split_spec_sub(mX, length)
    pXparts = split_spec_sub(pX, length)
    return mXparts, pXparts

def split_wav(x, length):
    n = len(x)/length
    parts = np.empty((n, length))
    for i in np.arange(0, n):
        parts[i] = x[i*length:(i+1)*length]
    return parts

def resize(x, tsec):
    tsamples = tsec * fs
    if(len(x) >= tsamples):
        return x[:tsamples]
    else:
        rep = tsamples/len(x) + 1
        return np.tile(x, rep)[:tsamples]
