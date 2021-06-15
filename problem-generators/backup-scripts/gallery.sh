#!/bin/bash

echo "finding solved problems in the directory, and link them to gallery/ directory"

rm -r gallery
mkdir -p gallery

# avoid duplicating the results
find -name '*problem*.png' | grep -v gallery | while read f
do
    ln -f -s ../$f gallery/$(echo ${f##./} | sed 's@/@-@g')
done
