__author__ = 'victor'

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FeedForwardNetwork, LinearLayer, FullConnection, IdentityConnection, TanhLayer

from dataset import *



# dataset
timeWidth = 5140 # num of samples to input to the net
mixer = MixedStream('piano', 'acapella', timeWidth)

# training
batchsize = 100
epochs = 1000


def build_net(width):
    net = FeedForwardNetwork()

    # layers
    net.addInputModule(LinearLayer(width, name='in'))
    net.addOutputModule(LinearLayer(width, name='out'))
    net.addModule(TanhLayer(100, name='h1'))
    net.addModule(TanhLayer(50, name='h2'))
    net.addModule(TanhLayer(100, name='h3'))

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


def train(mix, target):

    print 'preparing to train, netwidth=%d, batchsize=%d, epochs=%d' % (timeWidth, batchsize, epochs)
    net = build_net(timeWidth)
    # trainer = BackpropTrainer(net, learningrate=0.01, lrdecay=1, momentum=0.03, weightdecay=0)
    trainer = RPropMinusTrainer(net, learningrate=0.1, lrdecay=1, momentum=0.03, weightdecay=0)

    def train_batch(i):
        batch = SupervisedDataSet(timeWidth, timeWidth)
        begin = i * batchsize
        end = begin + batchsize
        for i in np.arange(begin, end):
            batch.addSample(mix[i], target[i])
        trainer.setData(batch)
        err = trainer.train()
        return err

    print 'training...'
    plot_cont(train_batch, epochs)

    # print 'saving net...'
    # err = trainer.train() # train an extra time just to get the final error
    # savenet(trainer.module, partlen, err)
    return net


def test(net, mix):
    print 'testing...'
    result = np.empty(timeWidth)
    for i in np.arange(500):
        netout = net.activate(mix[i])
        result = np.append(result, netout, axis=0)
    wavwrite(result, outputfile='output.wav')


net = train(mixer, mixer.stream1)
test(net, mixer)