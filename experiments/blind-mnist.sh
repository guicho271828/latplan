#!/bin/bash
./plan.py blind 'run_puzzle    ( "samples/mnist_puzzle33_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle"             ) )' |& tee $(dirname $0)/blind-mnist.log
