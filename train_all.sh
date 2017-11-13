#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

./strips.py conv puzzle learn_plot_dump mandrill 3 3 36 20000
./state_discriminator3.py samples/puzzle_mandrill_3_3_36_20000_conv/ learn_test
./action_autoencoder.py   samples/puzzle_mandrill_3_3_36_20000_conv/ learn_test
./action_discriminator.py samples/puzzle_mandrill_3_3_36_20000_conv/ learn_test

./strips.py conv puzzle learn_plot_dump mnist 3 3 36 20000
./state_discriminator3.py samples/puzzle_mnist_3_3_36_20000_conv/ learn_test
./action_autoencoder.py   samples/puzzle_mnist_3_3_36_20000_conv/ learn_test
./action_discriminator.py samples/puzzle_mnist_3_3_36_20000_conv/ learn_test

./strips.py conv puzzle learn_plot_dump spider 3 3 36 20000
./state_discriminator3.py samples/puzzle_spider_3_3_36_20000_conv/ learn_test
./action_autoencoder.py   samples/puzzle_spider_3_3_36_20000_conv/ learn_test
./action_discriminator.py samples/puzzle_spider_3_3_36_20000_conv/ learn_test

./strips.py conv lightsout learn_plot_dump digital 4 36 20000
./state_discriminator3.py samples/lightsout_digital_4_36_20000_conv/ learn_test
./action_autoencoder.py   samples/lightsout_digital_4_36_20000_conv/ learn_test
./action_discriminator.py samples/lightsout_digital_4_36_20000_conv/ learn_test

./strips.py conv lightsout learn_plot_dump twisted 4 36 20000
./state_discriminator3.py samples/lightsout_twisted_4_36_20000_conv/ learn_test
./action_autoencoder.py   samples/lightsout_twisted_4_36_20000_conv/ learn_test
./action_discriminator.py samples/lightsout_twisted_4_36_20000_conv/ learn_test

./strips.py conv hanoi learn_plot_dump 4 3 36 81
# ./state_discriminator3.py samples/hanoi_4_3_36_81_conv/ learn_test
# ./action_autoencoder.py   samples/hanoi_4_3_36_81_conv/ learn_test
# ./action_discriminator.py samples/hanoi_4_3_36_81_conv/ learn_test

