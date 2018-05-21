#!/bin/bash

## uncompressed sasp
# for sasp in $(find -name "*.sasp")
# do
#     if [ -s $sasp ]
#     then
#         if ! [[ "end_CG" == "$(tail -n 1 $sasp)" ]]
#         then
#             echo $sasp
#         fi
#     fi
# done

for sasp in $(find -name "*.sasp.gz")
do
    if [ $(du $sasp | cut -f 1) == 0 ]
    then
        echo $sasp
        # there may be false positive,
        # but they are small so ok to rerun anyways
    fi
done
