from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FullConnection, TanhLayer, RecurrentNetwork, LSTMLayer

from datasource import *


__author__ = 'victor'



# dataset
timeWidth = 1 # num of spectrogram time steps to input to the net each time
fourrier = Fourrier()
specMix = MixedSpectrumStream('piano', 'acapella', timeWidth, fourrier)
specTar = specMix.subStream1()
normMix = NormSpecStream(specMix)
normTar = NormSpecStream(specTar)
flatMix = FlatStream(normMix)
flatTarget = FlatStream(normTar)
pca = objread('pca')
pcaMix = PcaStream(flatMix, pca, 6)
pcaTarget = PcaStream(flatTarget, pca, 6)

# training
batchsize = 100
epochs = 1000
sampleShape = specMix.shape
netwidth = pcaMix.width # num of units in the input and output layers (magnitudes and phases)


def build_net(width):
    # net = FeedForwardNetwork()
    net = RecurrentNetwork()

    # layers
    net.addInputModule(TanhLayer(width, name='in'))
    net.addOutputModule(TanhLayer(width, name='out'))
    net.addModule(TanhLayer(50, name='h1'))
    net.addModule(TanhLayer(20, name='h2'))
    net.addModule(LSTMLayer(20, name='h2*'))
    net.addModule(TanhLayer(20, name='h3'))

    # connections
    net.addConnection(FullConnection(net['in'], net['h1']))
    net.addConnection(FullConnection(net['h1'], net['h2']))
    net.addConnection(FullConnection(net['h1'], net['h2*']))
    # net.addConnection(FullConnection(net['h1'], net['h3']))
    # net.addConnection(FullConnection(net['h1'], net['out']))
    net.addConnection(FullConnection(net['h2'], net['h3']))
    net.addConnection(FullConnection(net['h2*'], net['h3']))
    # net.addConnection(FullConnection(net['h2'], net['out']))
    net.addConnection(FullConnection(net['h3'], net['out']))
    # net.addConnection(IdentityConnection(net['in'], net['out']))

    # net.addRecurrentConnection(FullConnection(net['h1'], net['h1']))
    # net.addRecurrentConnection(FullConnection(net['h2'], net['h2']))

    net.sortModules()
    return net


def train(mixStream, targetStream):
    randomOffset = Random().randint(0, 1000) # randomly start on different data
    print 'preparing to train, netwidth=%d, batchsize=%d, epochs=%d, offset=%d' % (netwidth, batchsize, epochs, randomOffset)
    net = build_net(netwidth)
    trainer = RPropMinusTrainer(net, batchlearning=True, learningrate=0.01, lrdecay=1, momentum=0.1, weightdecay=0)

    def train_batch(i):
        batch = SupervisedDataSet(netwidth, netwidth)
        begin = randomOffset + i * batchsize
        end = begin + batchsize
        for j in np.arange(begin, end):
            batch.addSample(mixStream[j], targetStream[j])
        trainer.setData(batch)
        err = trainer.train()
        return err

    print 'training...'
    plot_cont(train_batch, epochs)

    # print 'saving net...'
    # err = trainer.train() # train an extra time just to get the final error
    # savenet(trainer.module, netwidth, err)
    return net


def test(net, mixStream):
    print 'testing...'
    mXresult = np.empty(sampleShape)
    pXresult = np.empty(sampleShape)
    for i in np.arange(500):
        netout = net.activate(mixStream[i])
        part = pcaMix.undo(netout)
        mXpart, pXpart = flatMix.unflatten(part)
        mXpart = normMix.unnorm(mXpart)
        mXresult = np.append(mXresult, mXpart, axis=0)
        pXresult = np.append(pXresult, pXpart, axis=0)
    fourrier.write(mXresult, pXresult)


net = train(pcaMix, pcaTarget)
test(net, pcaMix)
