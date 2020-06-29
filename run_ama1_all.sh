#!/bin/bash -x

ulimit -v 16000000000

trap exit SIGINT

probdir=problem-instances

#### foolproof check

# ensuring if the system is built correctly
(
    make -C lisp
    git submodule update --init --recursive
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
)

chmod -R +w $probdir            # in the weird case this happens

#### job submission

proj=$(date +%Y%m%d%H%M)ama1
command="jbsub -mem 128g -queue x86_24h -proj $proj task"
export PYTHONUNBUFFERED=1
export SHELL=/bin/bash
task (){
    base=$2/ama1_$(basename $1)_$4_$3
    outfile=$base.log
    errfile=$base.err
    trap "cat $outfile; cat $errfile >&2" RETURN
    ./ama1-planner.py \
        $@ \
        1> $outfile \
        2> $errfile
}
export -f task

parallel $command \
         ::: samples/puzzle*mnist* \
         ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
         ::: blind ::: all_actions actions

parallel $command \
         ::: samples/puzzle*mandrill* \
         ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
         ::: blind ::: all_actions actions


parallel $command \
         ::: samples/puzzle*spider* \
         ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
         ::: blind ::: all_actions actions


parallel $command \
         ::: samples/lightsout*digital* \
         ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
         ::: blind ::: all_actions actions


parallel $command \
         ::: samples/lightsout*twisted* \
         ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
         ::: blind ::: all_actions actions


# parallel --no-notice \
#          $command \
#          ::: samples/hanoi* \
#          ::: $probdir/*/latplan.puzzles.hanoi/* \
#          ::: blind ::: all_actions actions


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
