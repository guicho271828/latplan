#!/bin/bash

ulimit -v 16000000000

trap exit SIGINT

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


