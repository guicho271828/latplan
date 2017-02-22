#!/bin/bash
./plan.py mands 'run_puzzle ( "samples/mandrill_puzzle33_fc2" ,"fc2", import_module("puzzles.mandrill_puzzle" ) )' |& tee $(dirname $0)/mands-mandrill.log
