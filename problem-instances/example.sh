#!/bin/bash

trap exit SIGINT

./generate.py 7 5 puzzle mandrill 3 3
./generate.py 7 5 puzzle spider   3 3
./generate.py 7 5 puzzle mnist    3 3

./generate.py 7 5 hanoi 7 4
./generate.py 7 5 lightsout digital 4
# ./generate.py 7 5 lightsout twisted 4
