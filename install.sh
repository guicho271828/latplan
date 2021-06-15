#!/bin/bash

git submodule update --init --recursive

(
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
)


which ros || (
    git clone -b release https://github.com/roswell/roswell.git
    cd roswell
    sh bootstrap
    ./configure
    make
    sudo make install
    ros setup
)

ros dynamic-space-size=8000 install numcl arrival eazy-gnuplot magicffi dataloader

make -j 1 -C lisp

./setup.py install

./download-dataset.sh
