__author__ = 'victor'

from scipy.signal import get_window

from util import *

smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft


class Fourrier:

    def __init__(self, N=512):
        self.N = N # fft size (window + zero padding)
        self.M = N-1 # window size
        self.H = (self.M+1)/2 # stft hop size
        self.w = get_window("hamming", self.M)
        self.freqrange = N / 2 + 1 # dividing by 2 bc dft is mirrored. idk why the +1 though.
        # X time size ~ len x / hop size
        # X freq size ~ fft size / 2

    def analysis(self, x):
        return stft.stftAnal(x, fs, self.w, self.N, self.H)

    def synth(self, mX, pX):
        return stft.stftSynth(mX, pX, self.M, self.H)

    def write(self, mX, pX, outputfile='output.wav'):
        file = output_dir + outputfile
        x = stft.stftSynth(mX, pX, self.M, self.H)
        uf.wavwrite(x, fs, file)