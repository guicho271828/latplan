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

# chmod -R +w $probdir            # in the weird case this happens

#### job submission

key=${1:-*}

suffix=planning
base=samples-planning
taskfile=benchmark

task (){
    out=$(echo ${1%%.pddl} | sed 's@/@_@g')
    base=$2/ama3_${out}_$3
    outfile=$base.log
    errfile=$base.err
    trap "cat $outfile; cat $errfile >&2" RETURN
    ./ama3-planner.py \
        $@ \
        1> $outfile \
        2> $errfile
}
export -f task

(
    parallel echo task \
             ::: $base/puzzle_mnist*$suffix/${key}.pddl \
             ::: $probdir/*/latplan.puzzles.puzzle_mnist/* \
             ::: blind gc lama lmcut mands

    parallel echo task \
             ::: $base/puzzle_mandrill*$suffix/${key}.pddl \
             ::: $probdir/*/latplan.puzzles.puzzle_mandrill/* \
             ::: blind gc lama lmcut mands

    parallel echo task \
             ::: $base/puzzle_spider*$suffix/${key}.pddl \
             ::: $probdir/*/latplan.puzzles.puzzle_spider/* \
             ::: blind gc lama lmcut mands

    parallel echo task \
             ::: $base/lightsout_digital*$suffix/${key}.pddl \
             ::: $probdir/*/latplan.puzzles.lightsout_digital/* \
             ::: blind gc lama lmcut mands

    parallel echo task \
             ::: $base/lightsout_twisted*$suffix/${key}.pddl \
             ::: $probdir/*/latplan.puzzles.lightsout_twisted/* \
             ::: blind gc lama lmcut mands

    # parallel echo task \
        #          ::: $base/hanoi*$suffix/${key}.pddl \
        #          ::: $probdir/*/latplan.puzzles.hanoi/* \
        #          ::: blind gc lama lmcut mands
) > $taskfile

num=$(wc -l $taskfile)
sort -R $taskfile -o $taskfile

# 129600
# 16 cores : 

export hours=1
export taskperhour=4                       # 15min each
export cores=8
export taskperjob=$(($hours*$taskperhour*$cores))
export mem=3
export memperjob=$(($mem*$cores))

mkdir -p $taskfile-split
rm $taskfile-split/*
split -l $taskperjob $taskfile $taskfile-split/$taskfile.

proj=$(date +%Y%m%d%H%M)ama3
submit="jbsub -cores $cores -mem ${memperjob}g -queue x86_${hours}h -proj $proj"
export PYTHONUNBUFFERED=1

batchtask (){
    set -x
    echo $SHELL
    /usr/bin/env
    cat $1 | parallel --keep-order -j $cores
}
export -f batchtask

export SHELL=/bin/bash
for sub in $taskfile-split/$taskfile.*
do
    $submit batchtask $sub
done
