#!/bin/bash

trap exit SIGINT

./strips.py conv puzzle learn_plot_dump mnist 3 3 36
./state_discriminator3.py samples/puzzle_mnist3336_conv/ learn &
./action_discriminator.py samples/puzzle_mnist3336_conv/ learn &
./action_autoencoder.py   samples/puzzle_mnist3336_conv/ learn &
wait
./trivial-planner.py samples/puzzle_mnist3336_conv/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/

parallel --eta --timeout 120 --joblog parallel.log "./trivial-planner.py samples/puzzle_mnist3336_conv/ {1} GBFSRec > {1}/conv36.log" ::: instances-4step-highAD2/latplan.puzzles.puzzle_mnist/*

parallel --eta --timeout 120 --joblog parallel.log "./trivial-planner.py samples/puzzle_mnist3336_aconv/ {1} GBFSRec > {1}/aconv36.log" ::: instances-4step-highAD2/latplan.puzzles.puzzle_mnist/*

./strips.py fc puzzle_mandrill learn_plot_dump 3 3
./strips.py fc puzzle_lenna learn_plot_dump 3 3
./strips.py fc puzzle_spider learn_plot_dump 3 3

./strips.py fc hanoi learn_plot_dump 3
./strips.py fc hanoi learn_plot_dump 4
./strips.py fc xhanoi learn_plot_dump 4

./strips.py fc digital_lightsout learn_plot_dump 4
./strips.py fc digital_lightsout_skewed learn_plot_dump 3


