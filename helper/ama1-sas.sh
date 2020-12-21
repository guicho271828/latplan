#!/bin/bash +x

# dirty hack because python subprocess sucks

trap "rm $3" ERR

echo $0 $@
start=`date +%s`
if ! [ -s $3 ]
then
    $(dirname $0)/../lisp/ama1-sas.bin -t $1 $(cat $2) | gzip > $3
    end=`date +%s`
    echo $((end-start)) > $3.time
fi
