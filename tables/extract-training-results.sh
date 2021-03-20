#!/bin/bash

# This script parses each training result and turn them into a CSV table.

# note: see jq manual for -c, -r, -S meaning

if [[ $# == 0 ]]
then
    echo "Usage: $0 samples/*/logs/*/" >&2
    exit
fi

# print the CSV header
header (){
    if [[ ! -s $1/performance.json ]]
    then
        echo "$1/performance.json does not exist or is an empty file!" 
        return 1
    fi
    if [[ ! -s $1/aux.json ]]
    then
        echo "$1/aux.json does not exist or is an empty file!"
        return 1
    fi

    paste -d, \
        <(
        # parsing performance.json
        jq -c -r -S '[paths(type != "array" and type != "object")|map(tostring)|join(".")] | @csv' \
           $1/performance.json
    ) <(
        # parsing aux.json
        # we should remove some domain-specific parameters in order to have a consistent number of columns
        # header
        jq -c -r -S '[.parameters|del(.["fc_width","fc_depth","disks","towers","track","size","type","width","height","picsize","mean","std"])|paths(type != "array" and type != "object")|map(tostring)|join(".")] | @csv' \
           $1/aux.json
    ) <(
        # parsing parameter_count
        echo parameter_count
    )
}

# print remaining data rows
fn (){
    if [[ ! -s $1/performance.json ]]
    then
        echo "$1/performance.json does not exist or is an empty file!"
        return 1
    fi
    if [[ ! -s $1/aux.json ]]
    then
        echo "$1/aux.json does not exist or is an empty file!" 
        return 1
   fi

    paste -d, \
        <(
        # parsing performance.json
        # values
        jq -c -r -S '[getpath(paths(type != "array" and type != "object"))] | @csv' \
           $1/performance.json
    ) <(
        # parsing aux.json
        # we should remove some domain-specific parameters in order to have a consistent number of columns
        # values
        jq -c -r -S '[.parameters|del(.["fc_width","fc_depth","disks","towers","track","size","type","width","height","picsize","mean","std"])|getpath(paths(type != "array" and type != "object"))] | @csv' \
           $1/aux.json
    ) <(
        # parsing parameter_count
        cat $1/parameter_count.json
    )
}

export SHELL=/bin/bash
export -f fn


header $1
parallel -k fn ::: $@
