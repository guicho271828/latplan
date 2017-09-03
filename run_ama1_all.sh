#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

# Note: AMA1 requires huge memory and runtime for preprocessing.
# For example:
# puzzle instances require 4.5GB per process / 2 hours,
# lightsout instances require 7.5GB per process / 4 hours on Xeon E5-2676 2.4 GHz.
# Each SAS+ file may become over 1GB.

# re: behavior --- The preprocessing results are precious. They are always
# unique for each problem, irregardless of heuristics. However, due to the huge
# memory requirement, it is inefficient to preprocess the same
# problem independently.
# 
# Therefore, when a process is preprocessing an instance, other
# instances solving the same instances are waited through a file lock.
# 
# Note that even when a fd-planner process is waiting, it consumes nearly 700MB
# for already loaded NN image.

# Desired usage of this script is "./run_ama1_all.sh | parallel -j <number of processes>"
# where the number should be adjusted for the resource capacity on your system.

parallel --dry-run --no-notice --joblog problem-instances/latplan.puzzles.puzzle_mnist.ama1.csv  \
         "./fd-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.ama1.log" \
         ::: puzzle_mnist_3_3_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.puzzle_mnist/* \
         ::: blind pdb

parallel --dry-run --no-notice --joblog problem-instances/latplan.puzzles.puzzle_mandrill.ama1.csv   \
         "./fd-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.ama1.log" \
         ::: puzzle_mandrill_3_3_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.puzzle_mandrill/* \
         ::: blind pdb


parallel --dry-run --no-notice --joblog problem-instances/latplan.puzzles.puzzle_spider.ama1.csv   \
         "./fd-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.ama1.log" \
         ::: puzzle_spider_3_3_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.puzzle_spider/* \
         ::: blind pdb


parallel --dry-run --no-notice --joblog problem-instances/latplan.puzzles.lightsout_digital.ama1.csv   \
         "./fd-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.ama1.log" \
         ::: lightsout_digital_4_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.lightsout_digital/* \
         ::: blind pdb


parallel --dry-run --no-notice --joblog problem-instances/latplan.puzzles.lightsout_twisted.ama1.csv   \
         "./fd-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.ama1.log" \
         ::: lightsout_twisted_4_36_20000_conv \
         ::: problem-instances/*/latplan.puzzles.lightsout_twisted/* \
         ::: blind pdb


parallel --dry-run --no-notice --joblog problem-instances/latplan.puzzles.hanoi.ama1.csv   \
         "./fd-planner.py samples/{1} {2} {3} > {2}/{1}_{3}.ama1.log" \
         ::: hanoi_4_3_36_60_conv hanoi_4_3_36_81_conv \
         ::: problem-instances/*/latplan.puzzles.hanoi/* \
         ::: blind pdb

