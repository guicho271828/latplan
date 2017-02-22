#!/bin/bash
./plan.py mands 'run_hanoi ( "samples/hanoi3_fc2" ,"fc2", import_module("puzzles.hanoi" ) ,3)' |& tee $(dirname $0)/mands-hanoi3.log
