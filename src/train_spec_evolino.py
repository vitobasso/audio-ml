from pybrain.structure.modules.evolinonetwork import EvolinoNetwork
from pybrain.supervised.trainers.evolino import EvolinoTrainer

__author__ = 'victor'

from pybrain.datasets import SupervisedDataSet

from datasource import *




# dataset
timeWidth = 1 # num of spectrogram time steps to input to the net each time
fourrier = Fourrier()
mixSpec = MixedSpectrumStream('piano', 'acapella', timeWidth, fourrier)
targetSpec = mixSpec.subStream1()
flatMix = FlatStream(mixSpec)
flatTarget = FlatStream(targetSpec)

# training
batchsize = 100
epochs = 1000
sampleShape = mixSpec.shape
netwidth = flatMix.finalWidth # num of units in the input and output layers (magnitudes and phases)


def build_net(width):
    net = EvolinoNetwork(width, 40)
    net.sortModules()
    return net


def train(mixStream, targetStream):
    print 'preparing to train, netwidth=%d, batchsize=%d, epochs=%d' % (netwidth, batchsize, epochs)
    net = build_net(netwidth)
    trainer = EvolinoTrainer(net, verbosity=2)

    def train_batch(i):
        batch = SupervisedDataSet(netwidth, netwidth)
        begin = i * batchsize
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
        netout = net.activate(mixStream[i]) # TODO ??
        mXpart, pXpart = flatMix.unflatten(netout)
        mXresult = np.append(mXresult, mXpart, axis=0)
        pXresult = np.append(pXresult, pXpart, axis=0)
    mXrestored = unnormalize_static(mXresult)
    fourrier.write(mXrestored, pXresult, outputfile='output.wav')


net = train(flatMix, flatTarget)
test(net, flatMix)
