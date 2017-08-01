#!/bin/bash

trap exit SIGINT

./strips.py fc puzzle_mnist learn_plot_dump 3 3
./state_discriminator3.py samples/puzzle_mnist33_fc/ learn &
./action_discriminator.py samples/puzzle_mnist33_fc/ learn &
./action_autoencoder.py   samples/puzzle_mnist33_fc/ learn &
wait
./trivial-planner.py samples/puzzle_mnist33_fc/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/
parallel --bar --eta --timeout 50 --joblog parallel.log "./trivial-planner.py samples/puzzle_mnist33_fc/ {1} > {1}/fc.log" ::: trivial-planner-instances/latplan.puzzles.puzzle_mnist/*

./strips.py conv puzzle_mnist learn_plot_dump 3 3
./state_discriminator3.py samples/puzzle_mnist33_conv/ learn &
./action_discriminator.py samples/puzzle_mnist33_conv/ learn &
./action_autoencoder.py   samples/puzzle_mnist33_conv/ learn &
wait
./trivial-planner.py samples/puzzle_mnist33_conv/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/

parallel --bar --eta --timeout 50 --joblog parallel.log "./trivial-planner.py samples/puzzle_mnist33_conv/ {1} > {1}/conv.log" ::: trivial-planner-instances/latplan.puzzles.puzzle_mnist/*

./strips.py fc puzzle_mandrill learn_plot_dump 3 3
./strips.py fc puzzle_lenna learn_plot_dump 3 3
./strips.py fc puzzle_spider learn_plot_dump 3 3

./strips.py fc hanoi learn_plot_dump 3
./strips.py fc hanoi learn_plot_dump 4
./strips.py fc xhanoi learn_plot_dump 4

./strips.py fc digital_lightsout learn_plot_dump 4
./strips.py fc digital_lightsout_skewed learn_plot_dump 3


