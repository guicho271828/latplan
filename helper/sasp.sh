#!/bin/bash

# dirty hack because python subprocess sucks

echo $0 $@
$(dirname $0)/../downward/builds/release64/bin/preprocess < $1 > $2
