__author__ = 'victor'

from scipy.signal import get_window

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import TanhLayer

from util import *

smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import stft
import utilFunctions as UF


# files
soundfile1 = smstools_home + "/sounds/singing-female.wav"
soundfile2 = smstools_home + "/sounds/cello-double.wav"

# stft
N = 512 # dft size (window + zero padding)
M = N-1 # window size
H = (M+1)/2 # stft hop size
w = get_window("hamming", M)
freqrange = N / 2 + 1 # dividing by 2 bc dft is mirrored. idk why the +1 though.

# dataset
partlength = 50

# training
epochs = 30
flatwidth = partlength * freqrange
netwidth = 2 * flatwidth


def show(fs, X):
    print X.shape
    plot_stft(fs, X, N, H)

def write(fs, mX, pX, outputfile='output.wav'):
    specwrite(fs, mX, pX, M, H, outputfile=outputfile)

def loadspec(soundfile, len=5):
    fs, x = UF.wavread(soundfile)
    x = resize(fs, x, len)
    w = get_window("hamming", M)
    mX, pX = stft.stftAnal(x, fs, w, N, H)
    # X time size ~ len x / hop size
    # X freq size ~ fft size / 2)
    return fs, x, mX, pX

def mix(fs1, x1, fs2, x2):
    assert fs1 == fs2
    xmix = np.add(0.5*x1, 0.5*x2)
    mXmix, pXmix = stft.stftAnal(xmix, fs1, w, N, H)
    return fs1, xmix, mXmix, pXmix

def flatten_sample(v1, v2):
    res1 = np.reshape(v1, flatwidth)
    res2 = np.reshape(v2, flatwidth)
    return np.append(res1, res2)

def unflatten_sample(v):
    flatm = v[:flatwidth]
    flatp = v[flatwidth:]
    mX = np.reshape(flatm, (partlength, freqrange))
    pX = np.reshape(flatp, (partlength, freqrange))
    return mX, pX


def train():

    # prepare dataset
    fs1, x1, mX1, pX1 = loadspec(soundfile1)
    fs2, x2, mX2, pX2 = loadspec(soundfile2)
    assert fs1 == fs2
    fsmix, xmix, mXmix, pXmix = mix(fs1, x1, fs2, x2)
    write(fsmix, mXmix, pXmix, outputfile='mix.wav')
    write(fs2, mX1, pX1, outputfile='target.wav')
    # play('mix.wav', sync=True)
    # show(fs2, mXmix)
    mX1parts, pX1parts = split(mX1, pX1, partlength)
    mXmixparts, pXmixparts = split(mXmix, pXmix, partlength)
    assert len(mX1parts) == len(mXmixparts) == len(pX1parts) == len(pXmixparts)
    nparts = len(mX1parts)

    # train
    net = buildNetwork(netwidth, 150, netwidth, bias=True, hiddenclass=TanhLayer)
    dataset = SupervisedDataSet(netwidth, netwidth)
    for i in np.arange(nparts):
        sample = flatten_sample(mXmixparts[i], pXmixparts[i])
        target = flatten_sample(mX1parts[i], pX1parts[i])
        dataset.addSample(sample, target)
    trainer = BackpropTrainer(net, dataset)
    plot_cont(trainer.train, epochs)

    return net, nparts, mXmixparts, pXmixparts, fs1


def test(net, nparts, mXparts, pXparts, fs):
    mXresult = np.empty((partlength, freqrange))
    pXresult = np.empty((partlength, freqrange))
    for i in np.arange(nparts):
        sample = flatten_sample(mXparts[i], pXparts[i])
        netout = net.activate(sample)
        mXpart, pXpart = unflatten_sample(netout)
        mXresult = np.append(mXresult, mXpart, axis=0)
        pXresult = np.append(pXresult, pXpart, axis=0)

    write(fs, mXresult, pXresult, outputfile='output.wav')


net, nparts, mXparts, pXparts, fs = train()
test(net, nparts, mXparts, pXparts, fs)