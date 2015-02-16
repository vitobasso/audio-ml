__author__ = 'victor'

import wave
import contextlib
from random import Random

from fourrier import *
from cache import LRUCache
from settings import SMSTOOLS_MODELS, SAMPLES_HOME


sys.path.append(SMSTOOLS_MODELS)
import utilFunctions as uf


expected_fmt = dict(rate=44100, swidth=2, ch=1, compr='not compressed')


def wavLength(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        return f.getnframes()


cache = LRUCache(3)
def readPart(file, begin, end):
    assert begin <= end
    x = cache.get(file)
    if x is None:
        print 'loading: ' + file
        fs, x = uf.wavread(file)
        cache.set(file, x)
    assert end <= len(x)
    return x[begin:end]


def mapFiles(dirpath):
    result = []
    for path, dirs, files in os.walk(dirpath):
        for f in files:
            fpath = dirpath + '/' + f
            length = wavLength(fpath)
            result.append((f, length))
    return result

def totalLength(map):
    total = 0
    for file, length in map:
        total += length
    return total

def buffer(iterable, fun, shape):
    buff = np.empty(shape)
    for i in iterable:
        v = np.reshape(fun(i), (-1, shape[1]))
        buff = np.append(buff, v, axis=0)
    return buff

def buffer2(iterable, fun, shape):
    buff1 = np.empty(shape)
    buff2 = np.empty(shape)
    for i in iterable:
        v1, v2 = fun(i)
        buff1 = np.append(buff1, v1, axis=0)
        buff2 = np.append(buff2, v2, axis=0)
    return buff1, buff2


class Stream:
    '''
    Reads a bunch of audio files as if they were a unique and endless stream.
    Iterates circularly, rearranging the files in a different order every round.

    padding: extra samples to load around each chunk. they overlap between samples
    '''

    def __init__(self, folderName, chunkSize, padding=0):
        self.path = SAMPLES_HOME + folderName + '/'
        self.originalMap = mapFiles(self.path)
        self.sampleLength = totalLength(self.originalMap)
        assert self.sampleLength > chunkSize, '%d > %d' % (self.sampleLength, chunkSize)
        self.chunkSize = chunkSize
        self.random = Random()
        self.padding = padding

    def _reorderedmap(self, turn):
        reorderedMap = self.originalMap[:]
        self.random.seed(turn)
        self.random.shuffle(reorderedMap)
        return reorderedMap

    def _spin(self, i):
        turn = i / self.sampleLength
        bounded_i = i % self.sampleLength
        return turn, bounded_i

    def _seek(self, i):
        turn, bounded_i = self._spin(i)
        map = self._reorderedmap(turn)

        current = 0
        for count, (fileName, fileSize) in enumerate(map):
            next = current + fileSize
            if bounded_i < next:
                indexInFile = bounded_i - current
                return turn, count, indexInFile
            current = next
        raise IndexError

    def _straightscan(self, turn, beginFile, beginIndex, endFile, endIndex):
        # print 'straightscan: %d (%d, %d) to (%d, %d)' % (turn, beginFile, beginIndex, endFile, endIndex)
        assert beginFile <= endFile
        map = self._reorderedmap(turn)
        result = np.array([])
        for ifile in range(beginFile, endFile+1):
            fname, fsize = map[ifile]
            fpath = self.path + fname
            if(ifile == beginFile == endFile): # only one file
                part = readPart(fpath, beginIndex, endIndex)
            elif(ifile == beginFile): # first file
                part = readPart(fpath, beginIndex, fsize)
            elif(ifile == endFile): # last file
                part = readPart(fpath, 0, endIndex)
            else: # files in the middle
                part = readPart(fpath, 0, fsize)
            result = np.append(result, part)

        return result

    def _cyclescan(self, begin, end):
        # guaranteed to cycle at most once, since chunkSize <= totalLength
        assert begin < end
        beginTurn, beginFile, beginIndex = self._seek(begin)
        endTurn, endFile, endIndex = self._seek(end)
        # print 'cyclescan: %d (%d, %d, %d) to %d (%d, %d, %d)' % (begin, beginTurn, beginFile, beginIndex, end, endTurn, endFile, endIndex)
        assert beginTurn <= endTurn

        if beginTurn == endTurn:
            # straight
            return self._straightscan(beginTurn, beginFile, beginIndex, endFile, endIndex)
        else:
            # cyclic
            map = self._reorderedmap(beginTurn)
            lastFile = len(map) - 1
            lastFileName, lastFileLen = map[lastFile]
            result1stTurn = self._straightscan(beginTurn, beginFile, beginIndex, lastFile, lastFileLen)
            result2ndTurn = self._straightscan(endTurn, 0, 0, endFile, endIndex)
            return np.append(result1stTurn, result2ndTurn)

    def __getitem__(self, i):
        begin = i * self.chunkSize
        end = begin + self.chunkSize
        begin -= self.padding
        end += self.padding
        return self._cyclescan(begin, end)


class MixedStream:

    def __init__(self, folderName1, folderName2, chunkSize, padding=0):
        self.stream1 = Stream(folderName1, chunkSize, padding=padding)
        self.stream2 = Stream(folderName2, chunkSize, padding=padding)
        self.chunkSize = chunkSize
        self.padding = padding

    def __getitem__(self, i):
        x1 = self.stream1.__getitem__(i)
        x2 = self.stream2.__getitem__(i)
        return np.add(0.5*x1, 0.5*x2)


class SpectrumStream:

    def __init__(self, rawStream, fourrier=Fourrier(512)):
        assert isinstance(rawStream, Stream) or isinstance(rawStream, MixedStream)
        self.fourrier = fourrier
        self.rawStream = rawStream
        self.chunkSize = self.rawStream.chunkSize / self.fourrier.H
        self.shape = (self.chunkSize, self.fourrier.freqRange)
        assert rawStream.padding == fourrier.H

    def __getitem__(self, i):
        x = self.rawStream.__getitem__(i)
        assert len(x) == (self.chunkSize + 2) * self.fourrier.H # 2 extra for padding
        mX, pX = self.fourrier.analysis(x)
        mX = mX[1:-1] # remove padding
        pX = pX[1:-1] # remove padding
        assert mX.shape == pX.shape == self.shape
        return mX, pX

    # util
    def concat(self, n):
        shape = (0, self.fourrier.freqRange)
        mX = np.empty(shape)
        pX = np.empty(shape)
        for i in range(n):
            mXi, pXi = self[i]
            mX = np.append(mX, mXi, axis=0)
            pX = np.append(pX, pXi, axis=0)
        return mX, pX



class MixedSpectrumStream(SpectrumStream):

    def __init__(self, folderName1, folderName2, chunkSize, fourrier=Fourrier(512)):
        rawChunkSize = chunkSize * fourrier.H
        raw = MixedStream(folderName1, folderName2, rawChunkSize, padding=fourrier.H) # padding avoids artifacts in the transition between chunks
        SpectrumStream.__init__(self, raw, fourrier)

    def subStream1(self):
        return SpectrumStream(self.rawStream.stream1, self.fourrier)

    def subStream2(self):
        return SpectrumStream(self.rawStream.stream2, self.fourrier)


class NormSpecStream:
    '''
    Normalizes spectral magnitudes
    '''

    def __init__(self, specStream, mean=-80, scale=21): # default values were pre-measured from data
        assert isinstance(specStream, SpectrumStream)
        self.specStream = specStream
        self.shape = specStream.shape
        self.fourrier = specStream.fourrier
        self.mean = mean
        self.scale = scale

    def __getitem__(self, i):
        mX, pX = self.specStream.__getitem__(i)
        mX = (mX - self.mean) / self.scale
        return mX, pX

    def unnorm(self, mX):
        return mX * self.scale + self.mean


    # util

    def unnormMany(self, mX):
        print 'NormSpecStream: un-normalizing %d samples ...' % len(mX)
        shape = (0, self.fourrier.freqRange)
        fun = lambda v: self.unnorm(v)
        return buffer(mX, fun, shape)

    def buffer(self, n, offset=0):
        print 'NormSpecStream: buffering %d samples, starting from %d ...' % (n, offset)
        shape = (0, self.fourrier.freqRange)
        fun = lambda i: self[i]
        return buffer2(range(offset, offset+n), fun, shape)


class FlatStream:
    '''
    Flattens the arrays from SpectrumStream to be input to the net
    Pre-processes the audio:
        - normalize mean and std
        - reduce dimensionality (pca)
    '''

    def __init__(self, specStream):
        assert isinstance(specStream, SpectrumStream) or isinstance(specStream, NormSpecStream)
        self.specStream = specStream
        (dim1, dim2) = self.specStream.shape
        self.width = 2 * dim1 * dim2

    def _flatten(self, v1, v2):
        res1 = np.reshape(v1, self.width/2)
        res2 = np.reshape(v2, self.width/2)
        return np.append(res1, res2)

    def _unflatten(self, v):
        (dim1, dim2) = self.specStream.shape
        halfwidth = dim1 * dim2
        assert len(v.shape) == 1
        assert v.shape[0] == 2 * halfwidth
        flatm = v[:halfwidth]
        flatp = v[halfwidth:]
        mX = np.reshape(flatm, self.specStream.shape)
        pX = np.reshape(flatp, self.specStream.shape)
        return mX, pX

    def __getitem__(self, i):
        mX, pX = self.specStream.__getitem__(i)
        return self._flatten(mX, pX)

    def unflatten(self, v):
        return self._unflatten(v)


    # util

    def unflattenMany(self, V):
        print 'FlatStream: unflattening %d samples ...' % len(V)
        shape = (0, self.specStream.fourrier.freqRange)
        fun = lambda v: self.unflatten(v)
        return buffer2(V, fun, shape)

    def buffer(self, n, offset=0):
        print 'FlatStream: buffering %d samples, starting from %d ...' % (n, offset)
        shape = (n, self.width)
        fun = lambda i: self[i]
        return buffer(range(offset, offset+n), fun, shape)


class PcaStream:
    '''
    Runs pca over flattened spectrum data
    '''

    def __init__(self, flatStream, pca, scale=1): # scale pca result to fit tanh units
        assert isinstance(flatStream, FlatStream)
        self.flatStream = flatStream
        self.pca = pca
        self.width = pca.n_components_
        self.scale = scale

    def __getitem__(self, i):
        x = self.flatStream.__getitem__(i)
        return self.pca.transform(x) / self.scale

    def _restore_whitened(self, x_transformed):
        '''
        not implemented in scikit's PCA
        http://stackoverflow.com/questions/23254700/inversing-pca-transform-with-sklearn-with-whiten-true
        '''
        singular_values_sq = 1. / (self.pca.components_ ** 2).sum(axis=1)
        return np.dot(x_transformed, singular_values_sq[:, np.newaxis] * self.pca.components_) + self.pca.mean_

    def restore(self, x):
        if self.pca.whiten:
            x_restored = self._restore_whitened(x)
        else:
            x_restored = self.pca.inverse_transform(x)
        return x_restored * self.scale


    # util

    def undoMany(self, V):
        print 'PcaStream: undoing %d samples ...' % len(V)
        shape = (0, self.flatStream.width)
        fun = lambda v: self.restore(v)
        return buffer(V, fun, shape)

    def buffer(self, n, offset=0):
        print 'PcaStream: buffering %d samples, starting from %d ...' % (n, offset)
        shape = (0, self.width)
        fun = lambda i: self[i]
        return buffer(range(offset, offset+n), fun, shape)
