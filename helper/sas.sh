#!/bin/bash +x

# dirty hack because python subprocess sucks

trap "rm $3" ERR

echo $0 $@
[ -s $3 ] || ( $(dirname $0)/../lisp/sas.bin -t $1 $(cat $2) | gzip > $3 )
