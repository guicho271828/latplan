#!/bin/bash

name=$1

if [ -z $name ]
then
    (
        echo "Usage: $0 [configuration]"
        echo "Example: $0 PU-BINARY-80-100"
    ) 1>&2
    exit 1
fi

(
    echo "dir acc tpr tnr f mae prob_bitwise"
    cat samples/*/$name.csv
) | column -t

