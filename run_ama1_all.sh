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
# Note that even when a ama1-planner process is waiting, it consumes nearly 700MB
# for already loaded NN image.

# Desired usage of this script is "./run_ama1_all.sh | parallel -j <number of processes>"
# where the number should be adjusted for the resource capacity on your system.

#### foolproof check

# ensuring if the system is built correctly
(
    make -C lisp
    git submodule update --init --recursive
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release64
)

# in the weird case this happens
chmod -R +w noise-0.6-0.12-ama1

#### job submission
proj=$(date +%Y%m%d%H%M)
common="-mem 128g -queue x86_24h -proj $proj"
dir=$(dirname $(dirname $(readlink -ef $0)))
export PYTHONPATH=$dir:$PYTHONPATH
export PYTHONUNBUFFERED=1

parallel --no-notice \
         "jbsub $common './ama1-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama1.log 2> {2}/{1/}_{3}.ama1.err'" \
         ::: $(ls -d samples/puzzle*mnist* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama1/*/latplan.puzzles.puzzle_mnist/* \
         ::: blind

parallel --no-notice \
         "jbsub $common './ama1-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama1.log 2> {2}/{1/}_{3}.ama1.err'" \
         ::: $(ls -d samples/puzzle*mandrill* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama1/*/latplan.puzzles.puzzle_mandrill/* \
         ::: blind


parallel --no-notice \
         "jbsub $common './ama1-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama1.log 2> {2}/{1/}_{3}.ama1.err'" \
         ::: $(ls -d samples/puzzle*spider* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama1/*/latplan.puzzles.puzzle_spider/* \
         ::: blind


parallel --no-notice \
         "jbsub $common './ama1-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama1.log 2> {2}/{1/}_{3}.ama1.err'" \
         ::: $(ls -d samples/lightsout*digital* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama1/*/latplan.puzzles.lightsout_digital/* \
         ::: blind


parallel --no-notice \
         "jbsub $common './ama1-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama1.log 2> {2}/{1/}_{3}.ama1.err'" \
         ::: $(ls -d samples/lightsout*twisted* | grep -v 1000 | grep -v convn ) \
         ::: noise-0.6-0.12-ama1/*/latplan.puzzles.lightsout_twisted/* \
         ::: blind


# parallel --no-notice \
#          "jbsub $common './ama1-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama1.log 2> {2}/{1/}_{3}.ama1.err'" \
#          ::: samples/hanoi* \
#          ::: noise-0.6-0.12-ama1/*/latplan.puzzles.hanoi/* \
#          ::: blind

