#!/bin/bash -x

set -e

trap exit SIGINT

export SHELL=/bin/bash
ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 16g -cores 1 -queue x86_1h -proj $proj"

make -C lisp -j 1

to-pddl (){
    cd $1
    parallel --keep-order 'echo {1}/best.model $(ls -v {2}/*/best.model)' ::: $(ls -dv PRECONDITION/*) :::+ $(ls -dv EFFECT/*) > models
    dsama to-pddl models dsama.pddl
}

export -f to-pddl

common="jbsub -mem 9g -cores 1 -queue x86_1h -proj $proj"
parallel $common ::: to-pddl ::: samples/*/
