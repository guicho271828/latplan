#!/bin/bash


for log in *.log ; do
    search=$(awk '/^Actual search time/{print $4}' < $log)
    exp=$(awk '/^Expanded [0-9]+ state/{print $2}' < $log)
    total=$(awk '/^Total time/{print $3}' < $log)
    echo $(basename $log) $search $exp $total
done
