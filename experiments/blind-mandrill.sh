#!/bin/bash
./plan.py blind 'run_puzzle    ( "samples/mandrill_puzzle33_fc2"      ,"fc2", import_module("puzzles.mandrill_puzzle"          ) )' |& tee $(dirname $0)/blind-mandrill.log
