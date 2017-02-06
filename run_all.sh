#!/bin/bash

trap exit SIGINT

# hanoi: finished.
# lightsout: finished.
# mnist, mandrill: 
#  mnist_puzzle mandrill_puzzle
# digital_lightsout 
for type in fc ; do
    for task in hanoi  ; do
        for mode in learn ; do
            ./strips.py $type $task $mode
            samples/sync.sh
        done
    done
done

# for type in fc fcg ; do
#     for task in random_mnist_puzzle ; do
#         for mode in dump ; do
#             ./strips.py $type $task $mode
#             samples/sync.sh
#         done
#     done
# done
