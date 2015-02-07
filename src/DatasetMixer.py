
__author__ = 'victor'

import os
import sys
import wave
import contextlib

import numpy as np

import util


smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import utilFunctions as uf



samplesroot = '../../_dependencies/sound-samples/'
expected_fmt = dict(rate=44100, swidth=2, ch=1, compr='not compressed')

def wavLength(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        return f.getnframes()

def readPart(file, begin, end):
    # TODO for efficiency, cache the wav in memory or use low level random access w/ the wave module
    fs, x = uf.wavread(file)
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


class Packet:
    '''
    Reads a bunch of audio files as if they were a unique stream
    '''

    def __init__(self, folderName, chunkSize):
        self.path = samplesroot + folderName + '/'
        self.map = mapFiles(self.path)
        self.length = totalLength(self.map) / chunkSize
        self.chunkSize = chunkSize

    def seek(self, i):
        current = 0
        for count, (fileName, fileSize) in enumerate(self.map):
            next = current + fileSize
            if i < next:
                indexInFile = i - current
                return count, indexInFile
            current = next
        raise IndexError

    def readChunk(self, i):
        begin = i * self.chunkSize
        end = begin + self.chunkSize
        first, beginIndex = self.seek(begin)
        last, endIndex = self.seek(end)

        chunk = np.array([])
        for ifile in range(first, last+1):
            fname, fsize = self.map[ifile]
            fpath = self.path + fname
            if(ifile == first == last):
                part = readPart(fpath, beginIndex, endIndex)
            elif(ifile == first):
                part = readPart(fpath, beginIndex, fsize)
            elif(ifile == last):
                part = readPart(fpath, 0, endIndex)
            else:
                part = readPart(fpath, 0, fsize)
            chunk = np.append(chunk, part)

        return chunk


class DatasetMixer:

    def __init__(self, folderName1, folderName2, chunkSize):
        self.packet1 = Packet(folderName1, chunkSize)
        self.packet2 = Packet(folderName2, chunkSize)
        self.length = min(self.packet1.length, self.packet2.length)

    def readChunk(self, i):
        x1 = self.packet1.readChunk(i)
        x2 = self.packet2.readChunk(i)
        return np.add(0.5*x1, 0.5*x2)


mixer = DatasetMixer('guitar', 'drums', 200000)
x = mixer.readChunk(10)
util.wavwrite(x)
util.play(sync=True)
