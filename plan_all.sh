#!/bin/bash

trap exit SIGINT

parallel ./plan.py ::: blind pdb \
         ::: \
         "run_hanoi     hanoi                    samples/hanoi_fc2                  " \
         "run_puzzle    mnist_puzzle             samples/mnist_puzzle33p_fc2        " \
         "run_puzzle    lenna_puzzle             samples/lenna_puzzle33p_fc2        " \
         "run_puzzle    mandrill_puzzle          samples/mandrill_puzzle33p_fc2     " \
         "run_lightsout digital_lightsout        samples/digital_lightsout_fc2      " \
         "run_lightsout digital_lightsout_skewed samples/digital_lightsout_skewed_fc"













