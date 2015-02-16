__author__ = 'victor'

from scipy.signal import get_window

from util import *

smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft


class Fourrier:

    def __init__(self, fft_size=512, zpad=0):
        assert isinstance(fft_size, int)
        self.fft_size = fft_size # fft size (window + zero padding)
        zpad_ammount = int(zpad * fft_size)
        self.window_size = fft_size - zpad_ammount - 1  # window size
        self.hop = (self.window_size+1)/2 # stft hop size
        self.window = get_window("hamming", self.window_size)
        self.freqRange = fft_size / 2 + 1 # dividing by 2 bc dft is mirrored. idk why the +1 though.
        # X time size ~ len x / hop size
        # X freq size ~ fft size / 2

    def analysis(self, x):
        mX, pX = stft.stftAnal(x, fs, self.window, self.fft_size, self.hop)
        pX = pX % (2 * np.pi) # undo phase unwrapping
        return mX, pX

    def synth(self, mX, pX):
        return stft.stftSynth(mX, pX, self.window_size, self.hop)


    #util

    def write(self, mX, pX, outputfile='output.wav'):
        file = OUTPUT_HOME + outputfile
        x = stft.stftSynth(mX, pX, self.window_size, self.hop)
        uf.wavwrite(x, fs, file)

    def plot(self, mX):
        plot_stft(mX, self.fft_size, self.hop)