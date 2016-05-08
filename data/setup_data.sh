#! /bin/bash
echo 'Downloading wav files'
mkdir wavs/
for i in {1..34}; do
    wget http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/audio/s$i.tar;
    tar -xf s$i.tar
    mv s1 wavs/s1
    rm s$i.tar
done

mkdir aligns/
echo 'Downloading align files'
for i in {1..34}; do
    wget http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/align/s$i.tar;
    tar -xf s$i.tar
    mv align aligns/s$i
    rm s$i.tar
done

echo 'Setting up CQT directory'
mkdir cqt_data
mkdir cqt_data/words
for i in {1..34}; do
    mkdir cqt_data/s$i
done
