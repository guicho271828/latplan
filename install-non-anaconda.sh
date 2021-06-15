#!/bin/bash
set -e

sudo apt install -y gnuplot parallel
sudo apt install -y mercurial g++ cmake make python flex bison g++-multilib

# https://github.com/roswell/roswell/wiki/1.-Installation
sudo apt -y install build-essential automake libcurl4-openssl-dev

# magicffi
sudo apt -y install libmagic-dev

# for result parsing
sudo apt -y install sqlite3 jq

sudo apt -y install python3-pip

pip3 install --user -r requirements.txt
