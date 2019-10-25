#!/bin/bash -x

set -e

trap exit SIGINT

export SHELL=/bin/bash
ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
common="jbsub -mem 16g -cores 1 -queue x86_12h -proj $proj"

make -C lisp -j 1

actionlearner (){
    lisp/domain-actionlearner.bin $1/actions.csv > $1/actionlearner.pddl 2>$1/actionlearner.err
}

export -f actionlearner

common="jbsub -mem 9g -cores 1 -queue x86_1h -proj $proj"
# parallel $common ::: actionlearner ::: samples/*/

dsama-effect (){
    dir=$1
    [ -e $dir/action+ids.csv ]       || (tr "[:blank:]" "\t" < $dir/actions.csv | paste - $dir/action_ids.csv > $dir/action+ids.csv)
    [ -e $dir/action+ids.fasl ]      || (dsama dump-tsv $dir/action+ids.csv $dir/action+ids.fasl )

    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    actions=$(jq .parameters.M _aae/aux.json)
    bits=$(jq .parameters.N aux.json)
    parallel -j 16 --verbose "dsama train-model action+ids.fasl \"(:effect {} $@)\"" \
             ::: $(seq $actions) \
             ::: binary-random-forest-classifier diff-random-forest-classifier \
             ::: :n-tree \
             ::: 5 \
             ::: :bagging-ratio \
             ::: 0.34 \
             ::: :max-depth \
             ::: 10
    true
}

dsama-precondition (){
    dir=$1
    [ -e $dir/action+ids.csv ]       || (tr "[:blank:]" "\t" < $dir/actions.csv | paste - $dir/action_ids.csv > $dir/action+ids.csv)
    [ -e $dir/action+ids.fasl ]      || (dsama dump-tsv $dir/action+ids.csv $dir/action+ids.fasl )

    echo $dir
    echo $(readlink -ef $dir)
    cd $dir
    shift
    actions=$(jq .parameters.M _aae/aux.json)
    parallel -j 16 --verbose "dsama train-model action+ids.fasl \"(:precondition {} $@)\"" \
             ::: $(seq $actions) \
             ::: binary-random-forest-classifier pu-random-forest-classifier \
             ::: :n-tree \
             ::: 5 \
             ::: :bagging-ratio \
             ::: 0.34 \
             ::: :max-depth \
             ::: 10
    true
}

export -f dsama-effect dsama-precondition

common="jbsub -mem 9g -cores 16 -queue x86_1h -proj $proj"

parallel $common dsama-effect \
         ::: samples/*_1000_*/

common="jbsub -mem 9g -cores 16 -queue x86_1h -proj $proj"

parallel $common dsama-precondition \
         ::: samples/*_1000_*/

