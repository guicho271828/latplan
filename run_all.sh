#!/bin/bash

trap exit SIGINT

./strips.py fc puzzle_mnist learn_dump 3 3
./strips.py fc puzzle_mandrill learn_dump 3 3
./strips.py fc puzzle_lenna learn_dump 3 3
./strips.py fc puzzle_spider learn_dump 3 3

./strips.py fc hanoi learn_dump 3
./strips.py fc hanoi learn_dump 4
./strips.py fc xhanoi learn_dump 4

./strips.py fc digital_lightsout learn_dump 4
./strips.py fc digital_lightsout_skewed learn_dump 3


