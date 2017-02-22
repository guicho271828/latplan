#!/bin/bash
./plan.py blind 'run_hanoi("samples/hanoi9_fc2/","fc2",import_module("puzzles.hanoi"),9)' |& tee $(dirname $0)/blind-hanoi9.log
