#!/bin/bash +x

# dirty hack because python subprocess sucks

trap "rm $2" ERR

echo $0 $@
start=`date +%s`
if ! [ -s $2 ]
then
    $(dirname $0)/../lisp/problem.bin $(cat $1) > $2
    end=`date +%s`
    echo $((end-start)) > $2.time
fi


