#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

# not so interesting
# parallel --files <<EOF
# 
# ./strips.py conv puzzle summary mandrill 3 3 36 20000
# ./action_autoencoder.py   samples/puzzle_mandrill_3_3_36_20000_conv/ test
# 
# ./strips.py conv puzzle summary mnist 3 3 36 20000
# ./action_autoencoder.py   samples/puzzle_mnist_3_3_36_20000_conv/ test
# 
# ./strips.py conv puzzle summary spider 3 3 36 20000
# ./action_autoencoder.py   samples/puzzle_spider_3_3_36_20000_conv/ test
# 
# ./strips.py conv lightsout summary digital 4 36 20000
# ./action_autoencoder.py   samples/lightsout_digital_4_36_20000_conv/ test
# 
# ./strips.py conv lightsout summary twisted 4 36 20000
# ./action_autoencoder.py   samples/lightsout_twisted_4_36_20000_conv/ test
# 
# ./strips.py conv hanoi summary 4 3 36 60
# ./action_autoencoder.py   samples/hanoi_4_3_36_60_conv/ test
# EOF


parallel --files <<EOF

./state_discriminator3.py samples/puzzle_mandrill_3_3_36_20000_conv/ test
./action_discriminator.py samples/puzzle_mandrill_3_3_36_20000_conv/ test

./state_discriminator3.py samples/puzzle_mnist_3_3_36_20000_conv/ test
./action_discriminator.py samples/puzzle_mnist_3_3_36_20000_conv/ test

./state_discriminator3.py samples/puzzle_spider_3_3_36_20000_conv/ test
./action_discriminator.py samples/puzzle_spider_3_3_36_20000_conv/ test

./state_discriminator3.py samples/lightsout_digital_4_36_20000_conv/ test
./action_discriminator.py samples/lightsout_digital_4_36_20000_conv/ test

./state_discriminator3.py samples/lightsout_twisted_4_36_20000_conv/ test
./action_discriminator.py samples/lightsout_twisted_4_36_20000_conv/ test

./state_discriminator3.py samples/hanoi_4_3_36_60_conv/ test
./action_discriminator.py samples/hanoi_4_3_36_60_conv/ test
EOF
