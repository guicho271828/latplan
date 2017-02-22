#!/bin/bash
./plan.py mands 'run_lightsout3 ( "samples/digital_lightsout_skewed3_fc" ,"fc", import_module("puzzles.digital_lightsout_skewed" ) )' |& tee $(dirname $0)/mands-lightsout-skewed3.log
