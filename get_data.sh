#!/bin/bash

if [ ! -d ./data ]; then
    mkdir ./data
elif [ -d ./data ]; then
    echo "data directory already exists"
fi

if [ -e ./data/SST-2.zip ]; then
    echo "SST-2.zip already exists"
elif [ ! -e ./data/SST-2.zip ]; then
    curl 'https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8' -o ./data/SST-2.zip
    cd data
    unzip SST-2.zip
fi