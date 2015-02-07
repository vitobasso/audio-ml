__author__ = 'victor'

from scipy.signal import get_window

from util import *

smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft



# stft
N = 512 # dft size (window + zero padding)
M = N-1 # window size
H = (M+1)/2 # stft hop size
w = get_window("hamming", M)
freqrange = N / 2 + 1 # dividing by 2 bc dft is mirrored. idk why the +1 though.


def fourier(x):
    return stft.stftAnal(x, fs, w, N, H)
