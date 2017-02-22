#!/bin/bash
./plan.py pdb 'run_puzzle("samples/mnist_puzzle33_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=1)' |& tee $(dirname $0)/pdb-mnist1.log
