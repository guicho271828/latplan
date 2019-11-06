#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

# Note: AMA3 requires huge memory and runtime for preprocessing.
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
# Note that even when a ama3-planner process is waiting, it consumes nearly 700MB
# for already loaded NN image.

# Desired usage of this script is "./run_ama3_all.sh | parallel -j <number of processes>"
# where the number should be adjusted for the resource capacity on your system.

#### foolproof check

# ensuring if the system is built correctly
(
    make -C lisp
    git submodule update --init --recursive
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release64
)

probdir=problem-instances
# in the weird case this happens
chmod -R +w $probdir

#### job submission

key=$1
mem=${2:-64g}

proj=$(date +%Y%m%d%H%M)

common=" -mem $mem -queue x86_1h -proj $proj"
dir=$(dirname $(dirname $(readlink -ef $0)))
export PYTHONPATH=$dir:$PYTHONPATH
export PYTHONUNBUFFERED=1
export PATH=VAL:$PATH

command="jbsub -hold $common 'helper/ama3-planner.sh {1} {2} {3}'"


parallel -j 1 --no-notice "$command" \
         ::: samples/puzzle*mnist*/${key}*.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
         ::: blind 

parallel -j 1 --no-notice "$command" \
         ::: samples/puzzle*mandrill*/${key}*.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
         ::: blind 

parallel -j 1 --no-notice "$command" \
         ::: samples/puzzle*spider*/${key}*.pddl \
         ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
         ::: blind 

parallel -j 1 --no-notice "$command" \
         ::: samples/lightsout*digital*/${key}*.pddl \
         ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
         ::: blind 

parallel -j 1 --no-notice "$command" \
         ::: samples/lightsout*twisted*/${key}*.pddl \
         ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
         ::: blind 

# parallel -j 1 --no-notice \
#          "jbsub $common './ama3-planner.py {1} {2} {3} > {2}/{1/}_{3}.ama3.log 2> {2}/{1/}_{3}.ama3.err'" \
#          ::: samples/hanoi* \
#          ::: $probdir/*/latplan.puzzles.hanoi/* \
#          ::: blind  \
         # ::: remlic-1-1-0 remlic-2-2-0 remlic-4-4-0  actionlearner rf-2-others1b-t rf-5-others1b-t rf-10-others1b-t


