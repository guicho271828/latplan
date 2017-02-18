#!/bin/bash
./plan.py pdb 'run_puzzle("samples/mnist_puzzle33p_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=5)' |& tee $(dirname $0)/pdb-mnist5.log
