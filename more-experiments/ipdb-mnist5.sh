#!/bin/bash
./plan.py ipdb 'run_puzzle("samples/mnist_puzzle33_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=5)' |& tee $(dirname $0)/ipdb-mnist5.log
