#!/bin/bash -x

ulimit -v 16000000000

trap exit SIGINT

export cycle=1
export noise=0.0
probdir=problem-instances-10min-noise$noise-cycle$cycle
traindir=samples
suffix=
domainfilename=domain
proj=$(date +%Y%m%d%H%M)ama3

# each Fast Downward uses maximum 8g memory; 8g * 8 process = 64g ; plus a slack for NN footprint
submit="jbsub -cores 8 -mem 80g -queue x86_6h -proj $proj"

#### foolproof check

# ensuring if the system is built correctly
(
    make -C lisp
    git submodule update --init --recursive
    cd downward
    ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
)

# chmod -R +w $probdir            # in the weird case this happens

#### job submission

task (){
    out=$(echo ${1%%.pddl} | sed 's@/@_@g')
    base=$2/ama3_${out}_$3
    outfile=${base}.log
    errfile=${base}.err
    if [ -f ${base}_problem.json ]
    then
        echo skipping $base
        return
    fi
    trap "cat $outfile; cat $errfile >&2" RETURN
    ./ama3-planner.py \
        $@ $cycle $noise \
        1> $outfile \
        2> $errfile
}
export -f task

# 8 cores per job.
# singe task is timed out by 10 min, therefore 6 tasks / hour
# since the queue is 6h, we can assign 36 tasks per core

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/puzzle_mnist*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/puzzle*-mnist*/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/puzzle_mandrill*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/puzzle-mandrill*/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/lightsout_digital*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/lightsout-digital*/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/lightsout_twisted*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/lightsout-twisted*/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/blocks*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/prob-*/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/sokoban*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/sokoban*/[0-9]* \
                            ::: blind lama lmcut mands



./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/hanoi_4_4*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/hanoi-4-4/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/hanoi_3_9*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/hanoi-3-9/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/hanoi_4_9*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/hanoi-4-9/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 36 $submit -- task \
                            ::: $traindir/hanoi_5_9*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/hanoi-5-9/[0-9]* \
                            ::: blind lama lmcut mands



