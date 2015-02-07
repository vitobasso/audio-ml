__author__ = 'victor'

from pybrain.datasets import SupervisedDataSet

from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FeedForwardNetwork, LinearLayer, FullConnection, TanhLayer

from dataset import *



# dataset
speclen = 10 # num of spectrogram columns to input to the net
fourrier = Fourrier(512)
freqrange = fourrier.freqrange
rawlen = speclen * fourrier.H

# training
epochs = 100
flatwidth = speclen * freqrange
netwidth = 2 * flatwidth # num of units in the input and output layers (magnitudes and phases)


def flatten(v1, v2):
    res1 = np.reshape(v1, flatwidth)
    res2 = np.reshape(v2, flatwidth)
    return np.append(res1, res2)

def unflatten(v):
    flatm = v[:flatwidth]
    flatp = v[flatwidth:]
    mX = np.reshape(flatm, (speclen, freqrange))
    pX = np.reshape(flatp, (speclen, freqrange))
    return mX, pX


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


def train(nSamples, mixStream, targetStream):

    print 'preparing to train, nSamples=%d, netwidth=%d' % (nSamples, netwidth)
    net = build_net(netwidth)
    dataset = SupervisedDataSet(netwidth, netwidth)
    for i in np.arange(nSamples):
        sample = flatten(*mixStream.chunk(i))
        target = flatten(*targetStream.chunk(i))
        dataset.addSample(sample, target)
    # trainer = BackpropTrainer(net, dataset=dataset, learningrate=0.01, lrdecay=1, momentum=0.03, weightdecay=0)
    trainer = RPropMinusTrainer(net, dataset=dataset, learningrate=0.1, lrdecay=1, momentum=0.03, weightdecay=0)

    print 'training...'
    plot_cont(trainer.train, epochs)

    print 'saving net...'
    err = trainer.train() # train an extra time just to get the final error
    savenet(trainer.module, netwidth, err)
    return net


def test(net, nparts, mixStream):
    print 'testing...'
    mXresult = np.empty((speclen, freqrange))
    pXresult = np.empty((speclen, freqrange))
    for i in np.arange(nparts):
        sample = flatten(*mixStream.chunk(i))
        netout = net.activate(sample)
        mXpart, pXpart = unflatten(netout)
        mXresult = np.append(mXresult, mXpart, axis=0)
        pXresult = np.append(pXresult, pXpart, axis=0)
    mXrestored = unnormalize_static(mXresult)
    fourrier.write(mXrestored, pXresult, outputfile='output.wav')


mixer = PacketMixer('acapella', 'piano', rawlen)
mixSpec = SpectrumPacket(mixer, fourrier)
tarSpec = SpectrumPacket(mixer.packet1, fourrier)

nSamples = 100
net = train(nSamples, mixSpec, tarSpec)
test(net, nSamples, mixSpec)
