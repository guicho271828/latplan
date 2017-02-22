#!/bin/bash
./plan.py ipdb 'run_puzzle("samples/mnist_puzzle33p_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=4)' |& tee $(dirname $0)/ipdb-mnist4.log
