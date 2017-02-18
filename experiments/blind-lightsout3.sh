#!/bin/bash
./plan.py blind 'run_lightsout3 ( "samples/digital_lightsout_skewed3_fc" ,"fc",  import_module("puzzles.digital_lightsout_skewed" ) )' |& tee $(dirname $0)/blind-lightsout-skewed3.log
