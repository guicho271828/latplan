#!/bin/bash

trap exit SIGINT

./strips.py conv mnist_puzzle learn
./strips.py fc mnist_puzzle learn
./strips.py conv lenna_puzzle learn
./strips.py fc lenna_puzzle learn
./strips.py conv mandrill_puzzle learn
./strips.py fc mandrill_puzzle learn
./strips.py conv digital_lightsout learn
./strips.py fc digital_lightsout learn
./strips.py conv hanoi learn
./strips.py fc hanoi learn
