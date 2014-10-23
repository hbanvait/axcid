#!/bin/bash

#Clip Images
#cd Documents/imageclipper/build/bin
#./imageclipper -o /home/work/Documents/opencv1-haar-classifier-training_XingPed/Positive_images/%i
cd ../imageclipper/build/bin

export PATH=/usr/local/cuda-6.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-6.0/lib64:$LD_LIBRARY_PATH

# Clip images out of video
#./imageclipper /media/3637-3366/Videos/Capture-2.mp4   -o /home/work/Documents/Workspace/Training_XingPed/positive_images/%04i.jpg

# Clip images out of timed and extracted images
./imageclipper /media/3637-3366/Images/Positive_XingPed/   -o /home/work/Documents/Workspace/Training_35/positive_images/%i.jpg

# Clip images out of timed and extracted images in h w and all
./imageclipper /media/3637-3366/Images/Positive   -o /home/work/Documents/Workspace/Training_35/positive_images/%i_%04x_%04y_%04w_%04h.jpg
find positive_images/*_*_* -exec basename \{\} \; | perl -pe \
's/([^_]*_*_*_*.*)_0*(\d+)_0*(\d+)_0*(\d+)_0*(\d+)\.[^.]*$/$1 $2 $3 $4 $5\n/g' \
| tee clipping.txt

#./imageclipper /media/3637-3366/Videos/Capture\ 9\ \(8-8-2014\ 10-53\ AM\).mp4   -o /home/work/Documents/opencv1-haar-classifier-training_XingPed/positive_images/Capture_%i_%f.jpg

#./imageclipper /media/3637-3366/Videos/Capture\ 10\ \(8-8-2014\ 10-54\ AM\).mp4   -o /home/work/Documents/opencv1-haar-classifier-training_XingPed/positive_images/Capture_%i_%f.jpg

cd ../../../Training_XingPed
find ./positive_images -iname "*.jpg" > positives.txt

find ./negative_images -iname "*.jpg" > negatives.txt

perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
  "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
  -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 40 -h 20"

find ./samples -name '*.vec' > samples.txt
./mergevec samples.txt samples.vec

opencv_traincascade -data classifier -vec opencv_samples/opencv_samples.vec -bg negatives.txt\
  -numStages 3 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 800\
  -numNeg 1000 -mode ALL -precalcValBufSize 1024\
  -precalcIdxBufSize 1024 -featureType LBP

opencv_createsamples -vec opencv_samples/opencv_samples.vec -img positive_images/ -num 100 -bg negatives.txt  -maxxangle 0.6 -maxyangle 0 -maxzangle 0.3 -maxidev 100 -w 100 -h 50
#opencv_createsamples -vec opencv_samples/opencv_samples.vec -w 60 -h 30





