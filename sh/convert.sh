#!/bin/bash

# convert all sound files in folder tree to mono 44100 wav

files=$(find . -name "*.wav" -o -name "*.aif" -o -name "*.mp3" -o -name "*.flac")
for f in $files
do
    info="$(soxi -t "$f") $(soxi -c "$f") $(soxi -r "$f") $(soxi -b "$f")"
    if [ "$info" != "wav 1 44100 16" ]; then
        echo converting $f [$info]
        fnew=$(echo $f | sed -r s/\\.\\w+/_new.wav/)
        sox $f -c1 -r44100 -b16 $fnew
        rm $f
        mv $fnew $f
    fi
done
echo ok
