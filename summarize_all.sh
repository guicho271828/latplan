#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

./strips.py conv puzzle summary mandrill 3 3 36 20000
./strips.py conv puzzle summary mnist 3 3 36 20000
./strips.py conv puzzle summary spider 3 3 36 20000
./strips.py conv hanoi summary 7 4 36 10000
./strips.py conv lightsout summary digital 4 36 20000
./strips.py conv lightsout summary twisted 4 36 20000
