#!/bin/bash
./plan.py mands 'run_puzzle("samples/mnist_puzzle33_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=2)' |& tee $(dirname $0)/mands-mnist2.log
