__author__ = 'victor'
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

# plot signal and spectrogram
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


# plot spectrogram
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


# plot 1d-function updating continuously
def plot_cont(fun, xmax):
    y = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    def update(i):
        yi = fun()
        y.append(yi)
        x = np.arange(len(y))
        ax.clear()
        ax.plot(x, y)
        print i, ': ', yi

    a = anim.FuncAnimation(fig, update, frames=xmax, repeat=False)
    plt.show()