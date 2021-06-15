#!/bin/bash -x

# Generate PDDL domains for the training results stored in samples/ directory.

# Note: this script works on propositional action models dumped into csv. It
# works for both AMA3 and AMA4.

make -j 1 -C lisp

trap exit SIGINT

export SHELL=/bin/bash
ulimit -v 16000000000

export directory=${1:-samples}

run (){
    dir=$1
    lisp/ama3-domain.bin \
        $dir/available_actions.csv \
        $dir/action_add4.csv \
        $dir/action_del4.csv \
        $dir/action_pos4.csv \
        $dir/action_neg4.csv \
        > \
        $dir/domain.pddl
    lisp/ama3-domain.bin \
        $dir/available_actions.csv \
        $dir/action_add4.csv \
        $dir/action_del4.csv \
        > \
        $dir/domain-nopre.pddl
}

export -f run

common="parallel -j 1 --keep-order jbsub -mem 32g -cores 1 -queue x86_1h -proj pddl-ama3"

$common run ::: $directory/*/logs/*/
