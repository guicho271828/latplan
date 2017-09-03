#!/bin/bash

# dirty hack because python subprocess sucks

echo $0 $@
$(dirname $0)/../lisp/sas.bin -t $1 $(cat $2) > $3
