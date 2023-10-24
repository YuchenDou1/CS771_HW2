#!/bin/bash
# script for downloading the dataset

cd data
gdown 16GYHdSWS3iMYwMPv5FpeDZN2rH7PR0F2
# make sure that you have data.tar.gz in ./data
wget https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/train.txt
wget https://raw.githubusercontent.com/CSAILVision/miniplaces/master/data/val.txt

tar -xzf data.tar.gz
cd ..
