#!/bin/bash

# this experient does not make sense --- do not use!

export SHELL=/bin/bash

locality=${1:-0.1}

per_model_pre (){
    [ -d $1 ] || exit 1
    [ -d $2 ] || exit 1
    echo $(jq ".val.accuracy / 100" $1/best.json) $(jq ".val.accuracy / 100 " $2/best.json) $(readlink -ef $1/best.json) $(readlink -ef $2/best.json)
}

export -f per_model_pre

per_model_eff (){
    [ -d $1 ] || exit 1
    [ -d $2 ] || exit 1
    echo $(jq -s 'reduce .[] as $item (1; . * $item.val.accuracy / 100.0 )' $1/*/best.json) $(jq -s 'reduce .[] as $item (1; . * $item.val.accuracy / 100.0 )' $2/*/best.json) $(readlink -ef $1/best.json) $(readlink -ef $2/best.json)
}

export -f per_model_eff

trap "exit 1" INT 

parallel -j 24 per_model_pre ::: $(ls -vd samples/*_0.0/PRECONDITION/*) :::+ $(ls -vd samples/*_$locality/PRECONDITION/* ) > val.accuracy.precondition.comparedbylocality.$locality.csv
parallel -j 24 per_model_eff ::: $(ls -vd samples/*_0.0/EFFECT/*) :::+ $(ls -vd samples/*_$locality/EFFECT/* ) > val.accuracy.effect.comparedbylocality.$locality.csv

# per_model_eff samples/lightsout_HammingTransitionAE_digital_4_169_10000_0.0_0.0/EFFECT/10 samples/lightsout_HammingTransitionAE_digital_4_169_10000_0.0_0.1/EFFECT/1

# usage:
# ./compare-locality.sh 0.1 &
# ./compare-locality.sh 0.2 &
# ./compare-locality.sh 0.5 &
# ./compare-locality.sh 1.0 &
