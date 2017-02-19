#!/bin/bash
./plan.py blind 'run_hanoi("samples/hanoi5_fc2/","fc2",import_module("puzzles.hanoi"),5)' |& tee $(dirname $0)/blind-hanoi5.log
