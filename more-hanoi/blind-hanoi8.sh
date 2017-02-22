#!/bin/bash
./plan.py blind 'run_hanoi("samples/hanoi8_fc2/","fc2",import_module("puzzles.hanoi"),8)' |& tee $(dirname $0)/blind-hanoi8.log
