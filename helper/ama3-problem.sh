#!/bin/bash +x

# dirty hack because python subprocess sucks

ig=$1
out=$2

trap "rm $out" ERR

echo $0 $@
start=`date +%s`
if ! [ -s $out ]
then
    $(dirname $0)/../lisp/ama3-problem.bin $ig > $out
    end=`date +%s`
    echo $((end-start)) > $out.time
fi


