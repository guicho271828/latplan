#!/bin/bash

echo "finding solved problems in the directory, and link them to gallery/ directory"

gallery=$(dirname $(readlink -ef $0))/gallery.sh

for d in {vanilla,gaussian}/*/
do
    echo $d
    ( cd $d ; $gallery ) &
done

wait
