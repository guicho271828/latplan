#!/bin/bash
./plan.py mands 'run_hanoi ( "samples/xhanoi4_fc2" ,"fc2", import_module("puzzles.hanoi" ) ,4)' |& tee $(dirname $0)/mands-xhanoi4.log
