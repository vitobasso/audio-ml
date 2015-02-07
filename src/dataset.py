__author__ = 'victor'

import wave
import contextlib

from fourrier import *
from normalize import *
from cache import LRUCache


smstools_home = "../../_dependencies/sms-tools"
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), smstools_home + '/software/models/'))
import utilFunctions as uf


samplesroot = '../../_dependencies/sound-samples/'
expected_fmt = dict(rate=44100, swidth=2, ch=1, compr='not compressed')



def wavLength(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        return f.getnframes()


cache = LRUCache(3)
def readPart(file, begin, end):
    x = cache.get(file)
    if x is None:
        print 'loading: ' + file
        fs, x = uf.wavread(file)
        cache.set(file, x)
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

def wavstat(foldername):
    '''
    Compute statistics about the files
    '''
    folderpath = samplesroot + foldername
    allx = np.array([])
    for path, dirs, files in os.walk(folderpath):
        for f in files:
            fpath = folderpath + '/' + f
            fs, x = uf.wavread(fpath)
            allx = np.append(allx, x)
    print np.average(allx), np.std(allx), np.min(allx), np.max(allx)


#                   spectrogram     raw signal
#                   mean    std     mean    std
# guitar:           -72     23      -3e-05  0.02
# drums:            -70     22      8e-6    0.2
# piano:            -94     19      3e-6    0.08
# acapella:         -83     18      -6e-4   0.1
# violin:           -87     23      -8e-05  0.09



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

    def chunk(self, i):
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


class PacketMixer:

    def __init__(self, folderName1, folderName2, chunkSize):
        self.packet1 = Packet(folderName1, chunkSize)
        self.packet2 = Packet(folderName2, chunkSize)
        self.length = min(self.packet1.length, self.packet2.length)
        self.chunkSize = chunkSize

    def chunk(self, i):
        x1 = self.packet1.chunk(i)
        x2 = self.packet2.chunk(i)
        return np.add(0.5*x1, 0.5*x2)



class SpectrumPacket:

    def __init__(self, rawPacket, fourrier=Fourrier(512), normalized=True):
        self.fourrier = fourrier
        self.raw = rawPacket
        self.length = rawPacket.length
        self.chunkSize = rawPacket.chunkSize / self.fourrier.H
        self.normalized = normalized

    def chunk(self, i):
        x = self.raw.chunk(i)
        mX, pX = self.fourrier.analysis(x)
        if(self.normalized):
            mX = normalize_static(mX)
        return mX, pX


# mixer = PacketMixer('violin', 'piano', 5*fs)
# spect = SpectrumPacket(mixer)
#
# mX, pX = spect.chunk(0)
# mX = unnormalize_static(mX)
# spect.fourrier.write(mX, pX)
# play(sync=True)

