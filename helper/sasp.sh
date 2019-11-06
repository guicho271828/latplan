#!/bin/bash +x

# dirty hack because python subprocess sucks

trap "rm $2" ERR

echo $0 $@
start=`date +%s`
if ! [ -s $2 ]
then
    gunzip < $1 | $(dirname $0)/../downward/builds/release64/bin/preprocess | gzip > $2
    end=`date +%s`
    echo $((end-start)) > $2.time
fi
ln -b -s $(basename $2) $3
