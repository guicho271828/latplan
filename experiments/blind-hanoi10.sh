#!/bin/bash
./plan.py blind 'run_hanoi10   ( "samples/hanoi10_fc2"                 ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee $(dirname $0)/blind-hanoi10.log
