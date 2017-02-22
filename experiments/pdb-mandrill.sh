#!/bin/bash
./plan.py pdb 'run_puzzle ( "samples/mandrill_puzzle33_fc2" ,"fc2", import_module("puzzles.mandrill_puzzle" ) )' |& tee $(dirname $0)/pdb-mandrill.log
