__author__ = 'victor'

import os
import wave
import contextlib

samplesroot = '../../_dependencies/sound-samples/'
expected_fmt = dict(rate=44100, swidth=2, ch=1, compr='not compressed')


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

def load(dir):
    print 'loading files...'
    loadpath = samplesroot + dir
    for path, dirs, files in os.walk(loadpath):
        for f in files:
            print f
            validate_fmt(loadpath + '/' + f)


# wavinfo(root + 'violin/42953__freqman__gypsy-violin-variation.wav')
# wavinfo('/home/victor/dev/workspace/_dependencies/sms-tools/sounds/carnatic.wav')
# validate_fmt(root + 'violin/42953__freqman__gypsy-violin-variation.wav')

load('violin')