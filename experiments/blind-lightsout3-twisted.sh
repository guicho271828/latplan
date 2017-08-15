#!/bin/bash
./plan.py blind 'run_lightsout3 ( "samples/lightsout_twisted_3_fc" ,"fc", import_module("puzzles.lightsout_twisted" ) )' |& tee $(dirname $0)/blind-lightsout-twisted3.log
