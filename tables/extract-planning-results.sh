#!/bin/bash

# This script searches the directory for json files, then parses each file as a planning result and turn the results into a CSV table.

# note: see jq manual for the meaning of -c, -r, -S 

# note: jq manual is wrong about how to produce all paths to the leaf. It says paths(scalars), but since paths() requires a boolean filter,
# all false and null values get ignored.
# https://github.com/stedolan/jq/issues/2288

if [[ $# == 0 ]]
then
    echo "Usage: $0 directory" >&2
    exit
fi

# print the CSV header
header (){
    # we should remove some domain-specific parameters in order to have a consistent number of columns
    jq -c -r -S '[del(.parameters["fc_width","fc_depth","disks","towers","track","size","type","width","height","picsize","mean","std"],.statistics.total,.times)|paths(type != "array" and type != "object")|map(tostring)|join(".")] | @csv' \
       $1
}

# print remaining data rows
values (){
    # we should remove some domain-specific parameters in order to have a consistent number of columns
    jq -c -r -S '[del(.parameters["fc_width","fc_depth","disks","towers","track","size","type","width","height","picsize","mean","std"],.statistics.total,.times)|getpath(paths(type != "array" and type != "object"))] | @csv' \
       $1
}

export SHELL=/bin/bash
export -f header values

# debugging missing columns and random ordering
# parallel -k header ::: $(find $1 -name "*.json" | head -n 5)


header $(find $1 -name "*.json" | head -n 1)

# note: removing scene_tr/init.json and scene_tr/goal.json for blocksworld

find $1 -name "*.json" | \
    grep -v scene | \
    parallel -k values
