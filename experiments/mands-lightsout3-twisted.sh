#!/bin/bash
./plan.py mands 'run_lightsout3 ( "samples/lightsout_twisted_3_fc" ,"fc", import_module("puzzles.lightsout_twisted" ) )' |& tee $(dirname $0)/mands-lightsout-twisted3.log
