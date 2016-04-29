#!/bin/bash

SONGNAME="fly_me_to_the_moon"
# SONGNAME="chum"

# cd into yaafe folder to run yaafe scripts
cd ../yaafe-v0.64/src_python

yaafe.py -r 44100 -f "loudness_centroid: PerceptualSharpness blockSize=1024 stepSize=1024" $SONGNAME.wav
yaafe.py -r 44100 -f "loudness_spread: PerceptualSpread blockSize=1024 stepSize=1024" $SONGNAME.wav
yaafe.py -r 44100 -f "frequency_shape_statistics: SpectralShapeStatistics blockSize=1024 stepSize=1024" $SONGNAME.wav
yaafe.py -r 44100 -f "energy: Energy blockSize=1024 stepSize=1024" $SONGNAME.wav

# cd back to where we were 
cd ../../SongDataAPI

# dump newly created csv files into the newly created directory
DIRSUFFIX="_csv_files"
DIRNAME=$SONGNAME$DIRSUFFIX

mkdir $DIRNAME
mv ../yaafe-v0.64/src_python/$SONGNAME.wav.* $DIRNAME

# estract centroid/spread stats from csv files
python extract_stats.py $SONGNAME
