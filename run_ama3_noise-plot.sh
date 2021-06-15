#!/bin/bash -x


# This script just writes the result of unnormalize( normalize(inputs) + noise ) into files.
# See ama3-noise-plot.py .


ulimit -v 16000000000

trap exit SIGINT

probdir=problem-instances-noise-plot
traindir=samples
suffix=
domainfilename=domain
proj=$(date +%Y%m%d%H%M)ama3-noise-plot

submit="jbsub -cores 8 -mem 16g -queue x86_1h -proj $proj"

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
    ./ama3-noise-plot.py \
        $@ \
        1> $outfile \
        2> $errfile
}
export -f task

# 8 cores per job.
# singe task is timed out by 1 min, therefore 60 tasks / hour
# since the queue is 1h, we can assign 60 tasks per core

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $(ls $traindir/puzzle_mnist*$suffix/logs/*/$domainfilename.pddl | head -n 1) \
                            ::: $probdir/*/puzzle*-mnist*/[0-9]* \

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $(ls $traindir/puzzle_mandrill*$suffix/logs/*/$domainfilename.pddl | head -n 1) \
                            ::: $probdir/*/puzzle-mandrill*/[0-9]* \

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $(ls $traindir/lightsout_digital*$suffix/logs/*/$domainfilename.pddl | head -n 1) \
                            ::: $probdir/*/lightsout-digital*/[0-9]* \

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $(ls $traindir/lightsout_twisted*$suffix/logs/*/$domainfilename.pddl | head -n 1) \
                            ::: $probdir/*/lightsout-twisted*/[0-9]* \

# ./run-jobscheduler-parallel 8 60 $submit -- task \
#                             ::: $traindir/hanoi*$suffix/logs/*/$domainfilename.pddl \
#                             ::: $probdir/*/hanoi*/[0-9]* \
#                             ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $(ls $traindir/blocks*$suffix/logs/*/$domainfilename.pddl | head -n 1) \
                            ::: $probdir/*/prob-*/[0-9]* \

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $(ls $traindir/sokoban*$suffix/logs/*/$domainfilename.pddl | head -n 1) \
                            ::: $probdir/*/sokoban*/[0-9]* \

