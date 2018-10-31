#!/bin/bash

export SHELL=/bin/bash


per_model (){
    [ -d $1 ] || exit 1
    rm -f $1/best.json $1/best.model
    best=$(jq -r -s " max_by(.val.accuracy) | input_filename " $1/*.json )
    ln -s $(basename $best)                 $1/best.json
    ln -s $(basename ${best%%.json}.model)  $1/best.model    
}

export -f per_model

trap "exit 1" INT 

parallel -j 24 per_model ::: samples/*/PRECONDITION/*

for i in {0..120}
do
    parallel -j 24 per_model ::: samples/*/EFFECT/$i/*
done




