#!/bin/bash
./plan.py pdb 'run_lightsout3 ( "samples/lightsout_twisted_3_fc" ,"fc", import_module("puzzles.lightsout_twisted" ) )' |& tee $(dirname $0)/pdb-lightsout-twisted3.log
