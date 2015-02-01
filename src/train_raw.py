__author__ = 'victor'

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FeedForwardNetwork, LinearLayer, FullConnection, IdentityConnection, TanhLayer

from util import *


smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import utilFunctions as UF


# files
soundfile1 = smstools_home + "/sounds/singing-female.wav"
soundfile2 = smstools_home + "/sounds/cello-double.wav"

# dataset
trainsoundlen = 2 # duration in sec of the wav sounds loaded for training
partlen = 5140 # num of samples to input to the net

# training
epochs = 100


def show(fs, x):
    print fs, x.shape
    plot_wav(x)

def loadspec(soundfile, len):
    print 'loading wav:', soundfile, 'len:', len
    fs, x = UF.wavread(soundfile)
    x = resize(fs, x, len)
    return fs, x

def loadnorm(soundfile, len):
    fs, x = loadspec(soundfile, len)
    xnorm, avg, std = normalize_maxmin(x)
    return fs, xnorm, avg, std

def mix(fs1, x1, fs2, x2):
    assert fs1 == fs2
    xmix = np.add(0.5*x1, 0.5*x2)
    return fs1, xmix

def mixnorm(fs1, x1, fs2, x2):
    fs, xmix = mix(fs1, x1, fs2, x2)
    xnorm, avg, std = normalize_maxmin(xmix)
    return fs, xnorm, avg, std

def prepare_dataset(soundlen=trainsoundlen):

    # load and mix
    fs1, x1, avg1, std1 = loadnorm(soundfile1, len=soundlen)
    fs2, x2, avg2, std2 = loadnorm(soundfile2, len=soundlen)
    assert fs1 == fs2
    fs3, x3, avg3, std3 = mixnorm(fs1, x1, fs2, x2)
    # wavwrite(fs3, x3, outputfile='mix.wav')
    # wavwrite(fs2, x1, outputfile='target.wav')
    # play('mix.wav', sync=True)
    # show(fs2, x3)

    # split parts
    target_parts = split_wav(x1, partlen)
    mix_parts = split_wav(x3, partlen)
    assert len(target_parts) == len(mix_parts)
    nparts = len(target_parts)

    return fs1, nparts, mix_parts, target_parts, avg3, std3


def build_net(width):
    net = FeedForwardNetwork()

    # layers
    net.addInputModule(LinearLayer(width, name='in'))
    net.addOutputModule(LinearLayer(width, name='out'))
    net.addModule(TanhLayer(20, name='h1'))
    net.addModule(TanhLayer(50, name='h2'))
    net.addModule(TanhLayer(20, name='h3'))

    # connections
    net.addConnection(FullConnection(net['in'], net['h1']))
    net.addConnection(FullConnection(net['h1'], net['h2']))
    # net.addConnection(FullConnection(net['h1'], net['h3']))
    # net.addConnection(FullConnection(net['h1'], net['out']))
    net.addConnection(FullConnection(net['h2'], net['h3']))
    # net.addConnection(FullConnection(net['h2'], net['out']))
    net.addConnection(FullConnection(net['h3'], net['out']))
    net.addConnection(IdentityConnection(net['in'], net['out']))

    net.sortModules()
    return net


def train(nparts, sample, target):

    print 'preparing to train, nparts=%d, partlen=%d' % (nparts, partlen)
    net = build_net(partlen)
    dataset = SupervisedDataSet(partlen, partlen)
    for i in np.arange(nparts):
        dataset.addSample(sample[i], target[i])
    # trainer = BackpropTrainer(net, dataset=dataset, learningrate=0.01, lrdecay=1, momentum=0.03, weightdecay=0)
    trainer = RPropMinusTrainer(net, dataset=dataset, learningrate=0.1, lrdecay=1, momentum=0.03, weightdecay=0)

    print 'training...'
    plot_cont(trainer.train, epochs)

    print 'saving net...'
    err = trainer.train() # train an extra time just to get the final error
    savenet(trainer.module, partlen, err)
    return net


def test(net, fs, nparts, parts, avg, std):
    print 'testing...'
    result = np.empty(partlen)
    for i in np.arange(nparts):
        netout = net.activate(parts[i])
        result = np.append(result, netout, axis=0)
    xunnorm = unnormalize(result, avg, std)
    wavwrite(fs, xunnorm, outputfile='output.wav')


fs, nparts, sample, target, avg, std = prepare_dataset()
net = train(nparts, sample, target)
# fs, nparts, sample, target, avg, std = prepare_dataset(5)
# net = loadnet('net_1000_0.035637_2015-01-31T22:44:41')
test(net, fs, nparts, sample, avg, std)

# fs, x, avg, std = loadnorm(soundfile1, 5)
# show(fs, x)