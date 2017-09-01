#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

./strips.py conv puzzle learn_plot mandrill 3 3 36 5000
./strips.py conv puzzle learn_plot mnist 3 3 36 5000
./strips.py conv puzzle learn_plot spider 3 3 36 5000
./strips.py conv hanoi learn_plot 7 4 36 5000
./strips.py conv lightsout learn_plot digital 4 36 5000
./strips.py conv lightsout learn_plot twisted 4 36 5000
