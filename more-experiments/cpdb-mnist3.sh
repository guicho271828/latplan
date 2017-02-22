#!/bin/bash
./plan.py cpdb 'run_puzzle("samples/mnist_puzzle33_fc2","fc2",import_module("puzzles.mnist_puzzle"),init=3)' |& tee $(dirname $0)/cpdb-mnist3.log
