#!/bin/bash


domain=$1
# e.g. samples/puzzle...True/remlic.pddl
problem=$2
heur=$3

dir=$(dirname $(readlink -ef $0))

network=$(basename $(dirname $domain))
learner=$(basename $domain .pddl)

logname=$problem/${network}_${learner}_$heur.ama3

$dir/../ama3-planner.py $domain $problem $heur > $logname.log 2> $logname.err

