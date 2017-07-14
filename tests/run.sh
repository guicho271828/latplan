#!/bin/bash -x

set -e

parallel -j 4 python3 ::: ./counter_mnist.py ./counter_random_mnist.py ./digital_lightsout.py ./digital_lightsout_skewed.py ./hanoi.py ./puzzle_digital.py ./puzzle_lenna.py ./puzzle_mandrill.py ./puzzle_random_mnist.py ./puzzle_spider.py

