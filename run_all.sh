#!/bin/bash

trap exit SIGINT

./strips.py conv puzzle learn_plot_dump mnist 3 3 36 6500
./state_discriminator3.py samples/puzzle_mnist_3_3_36_6500_conv/ learn &
./action_autoencoder.py   samples/puzzle_mnist_3_3_36_6500_conv/ learn &
./action_discriminator.py samples/puzzle_mnist_3_3_36_6500_conv/ learn prepare_oae_PU3 &

wait
./trivial-planner.py samples/puzzle_mnist3336_conv/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/

./strips.py conv puzzle learn_plot_dump mnist 3 3 36 40000
./state_discriminator3.py samples/puzzle_mnist_3_3_36_40000_conv/ learn
./action_autoencoder.py   samples/puzzle_mnist_3_3_36_40000_conv/ learn
./action_discriminator.py samples/puzzle_mnist_3_3_36_40000_conv/ learn
parallel --eta --timeout 900 --joblog parallel.log "./trivial-planner.py samples/{1} {2} GBFSRec > {2}/{1}_GBFSRec.log" ::: puzzle_mnist_3_3_36_40000_conv ::: instances/latplan.puzzles.puzzle_mnist/*
parallel --eta --timeout 900 --joblog parallel.log "./trivial-planner.py samples/{1} {2} AstarRec > {2}/{1}_AstarRec.log" ::: puzzle_mnist_3_3_36_40000_conv ::: instances/latplan.puzzles.puzzle_mnist/*



parallel -j 1 ./strips.py {1} puzzle learn_plot mnist 3 3 36 {2} ::: conv aconv ::: 6500 13000 26000
parallel ./strips.py {1} puzzle dump mnist 3 3 36 {2} ::: conv aconv ::: 6500 13000 26000

parallel --eta --timeout 120 --joblog parallel.log "./trivial-planner.py samples/{1} {2} GBFSRec > {2}/{1}.log" ::: $(basename -a samples/puzzle_mnist*) ::: instances/latplan.puzzles.puzzle_mnist/*

parallel --eta --timeout 120 --joblog parallel.log "./trivial-planner.py samples/{1} {2} GBFSRec > {2}/{1}.log" ::: puzzle_mnist_3_3_36_6500_conv ::: instances/latplan.puzzles.puzzle_mnist/*

parallel -j 2 "./state_discriminator3.py samples/puzzle_mnist_3_3_36_{1}_{2}/ learn &> samples/puzzle_mnist_3_3_36_{1}_{2}/sd3.log" ::: 6500 13000 26000 ::: conv aconv
parallel -j 2 "./action_discriminator.py samples/puzzle_mnist_3_3_36_{1}_{2}/ learn &> samples/puzzle_mnist_3_3_36_{1}_{2}/ad.log"  ::: 6500 13000 26000 ::: conv aconv
parallel -j 2 "./action_autoencoder.py   samples/puzzle_mnist_3_3_36_{1}_{2}/ learn &> samples/puzzle_mnist_3_3_36_{1}_{2}/oae.log" ::: 6500 13000 26000 ::: conv aconv

./strips.py fc puzzle_mandrill learn_plot_dump 3 3
./strips.py fc puzzle_lenna learn_plot_dump 3 3
./strips.py fc puzzle_spider learn_plot_dump 3 3

./strips.py fc hanoi learn_plot_dump 3
./strips.py fc hanoi learn_plot_dump 4
./strips.py fc hanoi learn_plot_dump 7 4

./strips.py fc digital_lightsout learn_plot_dump 4
./strips.py fc digital_lightsout_skewed learn_plot_dump 3


