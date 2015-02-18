from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FullConnection, TanhLayer, RecurrentNetwork, LSTMLayer

from datasource import *


__author__ = 'victor'



# dataset
timeWidth = 1 # num of spectrogram time steps to input to the net each time
fourrier = Fourrier()
specMix = MixedSpectrumStream('acapella', 'guitar', timeWidth, fourrier)
specTar = specMix.subStream1()
flatMix = FlatStream(specMix)
flatTar = FlatStream(specTar)
scaler = objread('scaler_514')
stdMix = StandardStream(flatMix, scaler)
stdTar = StandardStream(flatTar, scaler)
pca = objread('pca_514_to_452_w')
pcaMix = PcaStream(stdMix, pca)
pcaTarget = PcaStream(stdTar, pca)

# training
batchsize = 100
epochs = 100000
sampleShape = specMix.shape
netwidth = pcaMix.width # num of units in the input and output layers (magnitudes and phases)


def build_net(width):
    # net = FeedForwardNetwork()
    net = RecurrentNetwork()

    # layers
    net.addInputModule(TanhLayer(width, name='in'))
    net.addOutputModule(TanhLayer(width, name='out'))
    net.addModule(TanhLayer(100, name='h1'))
    net.addModule(TanhLayer(50, name='h2'))
    net.addModule(LSTMLayer(50, name='h2*'))
    net.addModule(TanhLayer(100, name='h3'))

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

    net.addRecurrentConnection(FullConnection(net['h1'], net['h1']))
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
        t1 = time.time()
        for j in np.arange(begin, end):
            x = mixStream[j]
            y = targetStream[j]
            batch.addSample(x, y)
        dt = time.time() - t1
        print '\ttime spent on preprocessing: %.2fs' % (dt)
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
        x = net.activate(mixStream[i])
        x = pcaMix.restore(x)
        x = stdMix.unstandard(x)
        mX, pX = flatMix.unflatten(x)
        mXresult = np.append(mXresult, mX, axis=0)
        pXresult = np.append(pXresult, pX, axis=0)
    fourrier.plot(mXresult)
    fourrier.write(mXresult, pXresult)
    play(sync=True)


net = train(pcaMix, pcaTarget)
test(net, pcaMix)
