__author__ = 'victor'

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain import FeedForwardNetwork, LinearLayer, FullConnection, IdentityConnection, TanhLayer

from dataset import *




# dataset
trainsoundlen = 2 # duration in sec of the wav sounds loaded for training
partlen = 5140 # num of samples to input to the net

# training
nparts = 20
epochs = 100


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


def train(nparts, mix, target):

    print 'preparing to train, nparts=%d, partlen=%d' % (nparts, partlen)
    net = build_net(partlen)
    dataset = SupervisedDataSet(partlen, partlen)
    for i in np.arange(nparts):
        dataset.addSample(mix.chunk(i), target.chunk(i))
    # trainer = BackpropTrainer(net, dataset=dataset, learningrate=0.01, lrdecay=1, momentum=0.03, weightdecay=0)
    trainer = RPropMinusTrainer(net, dataset=dataset, learningrate=0.1, lrdecay=1, momentum=0.03, weightdecay=0)

    print 'training...'
    plot_cont(trainer.train, epochs)

    print 'saving net...'
    err = trainer.train() # train an extra time just to get the final error
    savenet(trainer.module, partlen, err)
    return net


def test(net, nparts, mix):
    print 'testing...'
    result = np.empty(partlen)
    for i in np.arange(nparts):
        netout = net.activate(mix.chunk(i))
        result = np.append(result, netout, axis=0)
    wavwrite(result, outputfile='output.wav')


mixer = PacketMixer('acapella', 'piano', partlen)
net = train(nparts, mixer, mixer.packet1)
test(net, nparts, mixer)