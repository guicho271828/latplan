#!/bin/bash -x


# This script reruns the image-based validations on the results stored in npz files.
# This script was used during the development for debugging the validator.


ulimit -v 16000000000

trap exit SIGINT

probdir=problem-instances-10min
traindir=samples
suffix=
domainfilename=domain
proj=$(date +%Y%m%d%H%M)ama3-revalidate

# each Fast Downward uses maximum 8g memory; 8g * 8 process = 64g ; plus a slack for NN footprint
submit="jbsub -cores 8 -mem 80g -queue x86_1h -proj $proj"

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
    outfile=$base.reval.log
    errfile=$base.reval.err
    if [ ! -f ${base}_problem.npz ]
    then
        echo skipping $base
        return
    fi
    trap "cat $outfile; cat $errfile >&2" RETURN
    ./ama3-revalidator.py \
        $@ \
        1> $outfile \
        2> $errfile
}
export -f task

# 8 cores per job.
# each task probably takes only below 1 min, therefore 60 tasks / hour
# since the queue is 1h, we can assign 60 tasks per core

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $traindir/puzzle_mnist*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/puzzle*-mnist*/[0-9]* \
                            ::: blind lama lmcut mands

./run-jobscheduler-parallel 8 60 $submit -- task \
                            ::: $traindir/puzzle_mandrill*$suffix/logs/*/$domainfilename.pddl \
                            ::: $probdir/*/puzzle-mandrill*/[0-9]* \
                            ::: blind lama lmcut mands

# ./run-jobscheduler-parallel 8 60 $submit -- task \
#                             ::: $traindir/lightsout_digital*$suffix/logs/*/$domainfilename.pddl \
#                             ::: $probdir/*/lightsout-digital*/[0-9]* \
#                             ::: blind lama lmcut mands
# 
# ./run-jobscheduler-parallel 8 60 $submit -- task \
#                             ::: $traindir/lightsout_twisted*$suffix/logs/*/$domainfilename.pddl \
#                             ::: $probdir/*/lightsout-twisted*/[0-9]* \
#                             ::: blind lama lmcut mands

# ./run-jobscheduler-parallel 8 60 $submit -- task \
#                             ::: $traindir/hanoi*$suffix/logs/*/$domainfilename.pddl \
#                             ::: $probdir/*/hanoi*/[0-9]* \
#                             ::: blind lama lmcut mands

# ./run-jobscheduler-parallel 8 60 $submit -- task \
#                             ::: $traindir/blocks*$suffix/logs/*/$domainfilename.pddl \
#                             ::: $probdir/*/prob-*/[0-9]* \
#                             ::: blind lama lmcut mands
# 
# ./run-jobscheduler-parallel 8 60 $submit -- task \
#                             ::: $traindir/sokoban*$suffix/logs/*/$domainfilename.pddl \
#                             ::: $probdir/*/sokoban*/[0-9]* \
#                             ::: blind lama lmcut mands

