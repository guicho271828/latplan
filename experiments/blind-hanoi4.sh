#!/bin/bash
./plan.py blind 'run_hanoi4    ( "samples/hanoi4_fc2"                  ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee $(dirname $0)/blind-hanoi4.log
