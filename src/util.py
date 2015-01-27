__author__ = 'victor'
import pygame
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft
import utilFunctions as UF

'''
plot signal and spectrogram
'''
def plot_stft(x, fs, mX, N, H):
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
plot spectrogram
'''
def plot_stft(fs, mX, N, H):
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
        print i, ': ', yi, '(', dt, ')'

    a = anim.FuncAnimation(fig, update, frames=xmax, repeat=False)
    plt.show()


'''
write sound file
optionally plays it
fs: frame rate (sec)
x: sound samples
'''
def writesound(fs, x, sync=False, outputfile='./output.wav', play=False):
    pygame.init()
    UF.wavwrite(x, fs, outputfile)
    if os.path.isfile(outputfile):
        if(play):
            sound = pygame.mixer.Sound(outputfile)
            ch = sound.play()
            if(sync):
                __wait_sound(ch)
    else:
        print "Output audio file not found", "The output audio file has not been computed yet"



'''
write sound file (given as spectrogram)
optionally plays it
fs: frame rate (sec)
mX: magnitude spectrum
pX: phase spectrum
M: window size
H: hop size
'''
def writespec(fs, mX, pX, M, H, sync=False, outputfile='./output.wav', play=False):
    x = stft.stftSynth(mX, pX, M, H)
    writesound(fs, x, sync, outputfile=outputfile, play=play)

def __wait_sound(ch):
    while ch.get_busy():
        time.sleep(1)