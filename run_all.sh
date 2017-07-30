#!/bin/bash

trap exit SIGINT

./strips.py fc puzzle_mnist learn_dump 3 3
./state_discriminator3.py samples/puzzle_mnist33_fc/ learn &
./action_discriminator.py samples/puzzle_mnist33_fc/ &
./action_autoencoder.py samples/puzzle_mnist33_fc/ &
wait
./trivial-planner.py samples/puzzle_mnist33_fc/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/

./strips.py conv puzzle_mnist learn_dump 3 3
./state_discriminator3.py samples/puzzle_mnist33_conv/ learn &
./action_discriminator.py samples/puzzle_mnist33_conv/ &
./action_autoencoder.py samples/puzzle_mnist33_conv/ &
wait
./trivial-planner.py samples/puzzle_mnist33_conv/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/

./strips.py fc puzzle_mandrill learn_dump 3 3
./strips.py fc puzzle_lenna learn_dump 3 3
./strips.py fc puzzle_spider learn_dump 3 3

./strips.py fc hanoi learn_dump 3
./strips.py fc hanoi learn_dump 4
./strips.py fc xhanoi learn_dump 4

./strips.py fc digital_lightsout learn_dump 4
./strips.py fc digital_lightsout_skewed learn_dump 3


