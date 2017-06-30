#!/bin/bash

trap exit SIGINT

./strips.py fc2 mnist_puzzle learn_dump 3 3
./strips.py fc2 mandrill_puzzle learn_dump 3 3

./strips.py fc2 hanoi learn_dump 3
./strips.py fc2 hanoi learn_dump 4
./strips.py fc2 xhanoi learn_dump 4

./strips.py fc2 digital_lightsout learn_dump 4
./strips.py fc digital_lightsout_skewed learn_dump 3


