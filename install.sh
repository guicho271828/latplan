#!/bin/bash
set -e

sudo apt install -y gnuplot parallel

sudo apt install -y mercurial g++ cmake make python flex bison g++-multilib
git submodule update --init --recursive
(
    git clone -b release-20.06.0 https://github.com/aibasel/downward.git
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
)


# https://github.com/roswell/roswell/wiki/1.-Installation
sudo apt -y install build-essential automake libcurl4-openssl-dev
git clone -b release https://github.com/roswell/roswell.git
(
    cd roswell
    sh bootstrap
    ./configure
    make
    sudo make install
    ros setup
)

sudo apt -y install libmagic-dev
ros dynamic-space-size=8000 install numcl arrival eazy-gnuplot
ros dynamic-space-size=8000 install guicho271828/magicffi guicho271828/dataloader

# for result parsing
sudo apt -y install sqlite3 jq

make -j 1 -C lisp

mkdir -p ~/.keras
cp keras-tf.json ~/.keras/keras.json

