#!/bin/bash

export probdir=${1:-problem-instances}


# ratio

fn (){
    key=$1
    jq -r ' [.s0, .s1, .s2, .s0-1, .t] | join(",") ' $probdir/*/*${key}/*/*-ood.json > ood-result-${key}.csv
}
export -f fn
SHELL=/bin/bash
# parallel fn ::: hanoi digital twisted mandrill mnist spider


jq -s -r ' group_by([.s0, .t ]) | map([.[0].s0-1, .[0].t,  length ] | join(",")) | .[] ' $probdir/*/*/*/*-ood.json > $probdir-ood-histogram-transitions.csv
jq -s -r ' group_by([.s0, .s1]) | map([.[0].s0-1, .[0].s1, length ] | join(",")) | .[] ' $probdir/*/*/*/*-ood.json > $probdir-ood-histogram-states.csv

