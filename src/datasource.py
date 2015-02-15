__author__ = 'victor'

import wave
import contextlib
from random import Random

from fourrier import *
from preprocess import normalize_static, pca_transform, pca_inverse, unnormalize_static, global_pca
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

def flatten(v1, v2, flatwidth):
    res1 = np.reshape(v1, flatwidth)
    res2 = np.reshape(v2, flatwidth)
    return np.append(res1, res2)

def unflatten(v, shape):
    (dim1, dim2) = shape
    flatwidth = dim1 * dim2
    flatm = v[:flatwidth]
    flatp = v[flatwidth:]
    mX = np.reshape(flatm, shape)
    pX = np.reshape(flatp, shape)
    return mX, pX


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
        assert mX.shape == pX.shape == (self.chunkSize, self.fourrier.freqRange)
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


class FlatStream:
    '''
    Flattens the arrays from SpectrumStream to be input to the net
    Pre-processes the audio:
        - normalize mean and std
        - reduce dimensionality (pca)
    '''

    def __init__(self, spectStream, preprocess=True):
        self.spectStream = spectStream
        self.chunkSize = self.spectStream.chunkSize
        (dim1, dim2) = self.spectStream.shape
        self.flatWidth = dim1 * dim2
        self.preprocess = preprocess
        if(preprocess):
            self.pca = global_pca
            self.finalWidth = len(global_pca.components_)

    def __getitem__(self, i):
        mX, pX = self.spectStream.__getitem__(i)
        if(self.preprocess):
            mX = normalize_static(mX)
        flatx = flatten(mX, pX, self.flatWidth)
        return pca_transform(flatx)

    def unflatten(self, v, undo_preprocessing=True):
        if(undo_preprocessing):
            v = pca_inverse(v)
        mX, pX = unflatten(v, self.spectStream.shape)
        if(undo_preprocessing):
            mX = unnormalize_static(mX)
        return mX, pX


    # util

    def unflattenMany(self, V, undo_preprocessing=True):
        shape = (0, self.spectStream.fourrier.freqRange)
        mX = np.empty(shape)
        pX = np.empty(shape)
        for v in V:
            mXi, pXi = self.unflatten(v, undo_preprocessing)
            mX = np.append(mX, mXi, axis=0)
            pX = np.append(pX, pXi, axis=0)
        return mX, pX

    def buffer(self, n, offset=0):
        print 'buffering %d samples, starting from %d...' % (n, offset)
        buff = np.array([])
        for i in range(offset, offset + n):
            buff = np.append(buff, self[i])
        buff = np.reshape(buff, (n, self.finalWidth))
        return buff

