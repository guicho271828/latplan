#!/bin/bash
./plan.py ipdb 'run_puzzle("samples/mnist_puzzle33_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=3)' |& tee $(dirname $0)/ipdb-mnist3.log
