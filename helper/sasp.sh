#!/bin/bash +x

# dirty hack because python subprocess sucks

trap "rm $2" ERR

echo $0 $@
[ -s $2 ] || ( gunzip < $1 | $(dirname $0)/../downward/builds/release64/bin/preprocess | gzip > $2 )
