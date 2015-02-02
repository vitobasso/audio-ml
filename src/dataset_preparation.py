__author__ = 'victor'

import os
import wave
from scipy.io.wavfile import read
import contextlib

import numpy as np


def wavinfo(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        print 'rate=%d, swidth=%d, ch=%d, compr=%s, frames=%d' % \
              (f.getframerate(), f.getsampwidth(), f.getnchannels(), f.getcompname(), f.getnframes())

def validate_fmt(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        assert f.getframerate() == expected_fmt['rate'], \
            'frame rate is %d. expected %d. file=%s' % (f.getframerate(),  expected_fmt['rate'], file)
        assert f.getsampwidth() == expected_fmt['swidth'], \
            'sample width is %d. expected %d. file=%s' % (f.getsampwidth(),  expected_fmt['swidth'], file)
        assert f.getnchannels() == expected_fmt['ch'], \
            'num channels is %d. expected %d. file=%s' % (f.getnchannels(),  expected_fmt['ch'], file)
        assert f.getcompname() == expected_fmt['compr'], \
            'compression name is %s. expected %s. file=%s' % (f.getcompname(),  expected_fmt['compr'], file)

def wavlength(file):
    with contextlib.closing(wave.open(file,'r')) as f:
        return f.getnframes()

def __wavread_norm_fmt(x):
    INT16_FAC = (2**15)-1
    INT32_FAC = (2**31)-1
    INT64_FAC = (2**63)-1
    norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}
    return np.float32(x)/norm_fact[x.dtype.name]

def wavread(file, begin, length):
    #TODO for efficiency, cache the wav in memory or use low level random access w/ the wave module
    validate_fmt(file)
    fs, x = read(file)
    x = __wavread_norm_fmt(x)
    return x[begin : begin + length]



samplesroot = '../../_dependencies/sound-samples/'
expected_fmt = dict(rate=44100, swidth=2, ch=1, compr='not compressed')

def summarize_files(dir):
    result = []
    loadpath = samplesroot + dir
    for path, dirs, files in os.walk(loadpath):
        for f in files:
            fpath = loadpath + '/' + f
            validate_fmt(fpath)
            length = wavlength(fpath)
            result.append((f, length))
    return result


snd1 = samplesroot + 'violin/42953__freqman__gypsy-violin-variation.wav'
snd2 = '/home/victor/dev/workspace/_dependencies/sms-tools/sounds/carnatic.wav'
# wavinfo(snd1)
# wavinfo(snd2)
# validate_fmt(snd1)
# load('violin')
# wavread(snd2, 0, 10)
summarize_files('violin')