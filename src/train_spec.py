__author__ = 'victor'

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FeedForwardNetwork, LinearLayer, FullConnection, TanhLayer, IdentityConnection

from dataset import *




# dataset
timeWidth = 10 # num of spectrogram columns to input to the net
fourrier = Fourrier(512)
mixSpec = MixedSpectrumStream('acapella', 'piano', timeWidth, fourrier)
targetSpec = mixSpec.subStream1()
flatMix = FlatStream(mixSpec)
flatTarget = FlatStream(targetSpec)

# training
nparts = 100
epochs = 100
sampleShape = mixSpec.shape
netwidth = 2 * flatMix.flatWidth # num of units in the input and output layers (magnitudes and phases)


def build_net(width):
    net = FeedForwardNetwork()

    # layers
    net.addInputModule(LinearLayer(width, name='in'))
    net.addOutputModule(LinearLayer(width, name='out'))
    net.addModule(TanhLayer(50, name='h1'))
    net.addModule(TanhLayer(20, name='h2'))
    # net.addModule(SigmoidLayer(10, name='h3'))

    # connections
    net.addConnection(FullConnection(net['in'], net['h1']))
    net.addConnection(FullConnection(net['h1'], net['h2']))
    # net.addConnection(FullConnection(net['h1'], net['h3']))
    net.addConnection(FullConnection(net['h1'], net['out']))
    # net.addConnection(FullConnection(net['h2'], net['h3']))
    net.addConnection(FullConnection(net['h2'], net['out']))
    # net.addConnection(FullConnection(net['h3'], net['out']))
    net.addConnection(IdentityConnection(net['in'], net['out']))

    net.sortModules()
    return net


def train(nparts, mixStream, targetStream):

    print 'preparing to train, nSamples=%d, netwidth=%d' % (nparts, netwidth)
    net = build_net(netwidth)
    dataset = SupervisedDataSet(netwidth, netwidth)
    for i in np.arange(nparts):
        dataset.addSample(mixStream[i], targetStream[i])
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
    mXresult = np.empty(sampleShape)
    pXresult = np.empty(sampleShape)
    for i in np.arange(nparts):
        netout = net.activate(mixStream[i])
        mXpart, pXpart = flatMix.unflatten(netout)
        mXresult = np.append(mXresult, mXpart, axis=0)
        pXresult = np.append(pXresult, pXpart, axis=0)
    mXrestored = unnormalize_static(mXresult)
    fourrier.write(mXrestored, pXresult, outputfile='output.wav')


net = train(nparts, flatMix, flatTarget)
test(net, nparts, flatMix)
