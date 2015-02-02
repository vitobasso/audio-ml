__author__ = 'victor'

from scipy.signal import get_window

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FeedForwardNetwork, LinearLayer, FullConnection, TanhLayer

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
trainsoundlen = 2 # duration in sec of the wav sounds loaded for training
partlen = 10 # num of spectrogram columns to input to the net

# training
epochs = 100
flatwidth = partlen * freqrange
netwidth = 2 * flatwidth # num of units in the input and output layers (magnitudes and phases)


def show(fs, X):
    print X.shape
    plot_stft(fs, X, N, H)

def write(fs, mX, pX, outputfile):
    specwrite(fs, mX, pX, M, H, outputfile=outputfile)

def loadspec(soundfile, len):
    print 'loading wav:', soundfile, 'len:', len
    fs, x = UF.wavread(soundfile)
    x = resize(fs, x, len)
    w = get_window("hamming", M)
    mX, pX = stft.stftAnal(x, fs, w, N, H)
    # X time size ~ len x / hop size
    # X freq size ~ fft size / 2)
    return fs, x, mX, pX

def loadnorm(soundfile, len):
    fs, x, mX, pX = loadspec(soundfile, len)
    mXnorm, avg, std = normalize_gauss(mX)
    return fs, x, mXnorm, pX, avg, std

def mix(fs1, x1, fs2, x2):
    assert fs1 == fs2
    xmix = np.add(0.5*x1, 0.5*x2)
    mXmix, pXmix = stft.stftAnal(xmix, fs1, w, N, H)
    return fs1, xmix, mXmix, pXmix

def mixnorm(fs1, x1, fs2, x2):
    fs, xmix, mXmix, pXmix = mix(fs1, x1, fs2, x2)
    mXnorm, avg, std = normalize_gauss(mXmix)
    return fs, xmix, mXnorm, pXmix, avg, std

def flatten_sample(v1, v2):
    res1 = np.reshape(v1, flatwidth)
    res2 = np.reshape(v2, flatwidth)
    return np.append(res1, res2)

def unflatten_sample(v):
    flatm = v[:flatwidth]
    flatp = v[flatwidth:]
    mX = np.reshape(flatm, (partlen, freqrange))
    pX = np.reshape(flatp, (partlen, freqrange))
    return mX, pX


def prepare_dataset(soundlen=trainsoundlen):

    # load and mix
    fs1, x1, mX1, pX1, avg1, std1 = loadnorm(soundfile1, len=soundlen)
    fs2, x2, mX2, pX2, avg2, std2 = loadnorm(soundfile2, len=soundlen)
    assert fs1 == fs2
    fs3, x3, mX3, pX3, avg3, std3 = mixnorm(fs1, x1, fs2, x2)
    # write(fs3, mX3, pX3, outputfile='mix.wav')
    # write(fs2, mX1, pX1, outputfile='target.wav')
    # play('mix.wav', sync=True)
    # show(fs2, mX3)

    # split parts
    mXtarget_parts, pXtarget_parts = split_spec(mX1, pX1, partlen)
    mXmix_parts, pXmix_parts = split_spec(mX3, pX3, partlen)
    assert len(mXtarget_parts) == len(mXmix_parts) == len(pXtarget_parts) == len(pXmix_parts)
    nparts = len(mXtarget_parts)

    return fs1, nparts, mXmix_parts, pXmix_parts, mXtarget_parts, pXtarget_parts, avg3, std3


def build_net(width):
    net = FeedForwardNetwork()

    # layers
    net.addInputModule(LinearLayer(width, name='in'))
    net.addOutputModule(LinearLayer(width, name='out'))
    net.addModule(TanhLayer(50, name='h1'))
    # net.addModule(TanhLayer(20, name='h2'))
    # net.addModule(SigmoidLayer(10, name='h3'))

    # connections
    net.addConnection(FullConnection(net['in'], net['h1']))
    # net.addConnection(FullConnection(net['h1'], net['h2']))
    # net.addConnection(FullConnection(net['h1'], net['h3']))
    net.addConnection(FullConnection(net['h1'], net['out']))
    # net.addConnection(FullConnection(net['h2'], net['h3']))
    # net.addConnection(FullConnection(net['h2'], net['out']))
    # net.addConnection(FullConnection(net['h3'], net['out']))
    # net.addConnection(IdentityConnection(net['in'], net['out']))

    net.sortModules()
    return net


def train(nparts, msample, psample, mtarget, ptarget):

    print 'preparing to train, nparts=%d, netwidth=%d' % (nparts, netwidth)
    net = build_net(netwidth)
    dataset = SupervisedDataSet(netwidth, netwidth)
    for i in np.arange(nparts):
        sample = flatten_sample(msample[i], psample[i])
        target = flatten_sample(mtarget[i], ptarget[i])
        dataset.addSample(sample, target)
    # trainer = BackpropTrainer(net, dataset=dataset, learningrate=0.01, lrdecay=1, momentum=0.03, weightdecay=0)
    trainer = RPropMinusTrainer(net, dataset=dataset, learningrate=0.1, lrdecay=1, momentum=0.03, weightdecay=0)

    print 'training...'
    plot_cont(trainer.train, epochs)

    print 'saving net...'
    err = trainer.train() # train an extra time just to get the final error
    savenet(trainer.module, netwidth, err)
    return net


def test(net, fs, nparts, mXparts, pXparts, avg, std):
    print 'testing...'
    mXresult = np.empty((partlen, freqrange))
    pXresult = np.empty((partlen, freqrange))
    for i in np.arange(nparts):
        sample = flatten_sample(mXparts[i], pXparts[i])
        netout = net.activate(sample)
        mXpart, pXpart = unflatten_sample(netout)
        mXresult = np.append(mXresult, mXpart, axis=0)
        pXresult = np.append(pXresult, pXpart, axis=0)
    mXunnorm = unnormalize(mXresult, avg, std)
    write(fs, mXunnorm, pXresult, outputfile='output.wav')


fs, nparts, msample, psample, mtarget, ptarget, avg, std = prepare_dataset()
net = train(nparts, msample, psample, mtarget, ptarget)
# fs, nparts, msample, psample, mtarget, ptarget, avg, std = prepare_dataset(5)
# net = loadnet('net_5140_3.156083_2015-01-31T22:48:39')
test(net, fs, nparts, msample, psample, avg, std)