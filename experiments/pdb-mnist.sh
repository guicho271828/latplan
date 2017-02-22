#!/bin/bash
./plan.py pdb 'run_puzzle    ( "samples/mnist_puzzle33_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle"             ) )' |& tee $(dirname $0)/pdb-mnist.log
