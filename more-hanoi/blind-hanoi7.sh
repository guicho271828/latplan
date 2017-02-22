#!/bin/bash
./plan.py blind 'run_hanoi("samples/hanoi7_fc2/","fc2",import_module("puzzles.hanoi"),7)' |& tee $(dirname $0)/blind-hanoi7.log
