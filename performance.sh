#!/bin/bash

export SHELL=/bin/bash

try (){
    ( $@ &> /dev/null ) && ( $@ )
}

export -f try

column (){
    f=$(mktemp) 
    parallel --keep-order \
             " try jq -e $1 {1}performance.json || try jq -e $1 {1}aux.json " \
             :::  $(ls -vd samples/*/) \
             > $f
    echo $f
}

export -f column

files=$(parallel --keep-order column ::: $@)

paste $files
rm $files

# usage:
# ./preview.sh .MSE.vanilla.test \
#              .inactive.both.test \
#              .hamming.test \
#              .parameters.zerosuppress \
#              .parameters.locality
