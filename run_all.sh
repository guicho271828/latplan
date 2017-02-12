#!/bin/bash

trap exit SIGINT

# gauss fc fc2 fcg fcg2 conv convg 
# for type in gauss ; do
#     for task in digital_lightsout ; do
#         for mode in learn ; do
#             ./strips.py $type $task $mode
#             samples/sync.sh
#         done
#     done
# done

# hanoi: finished.
# digital_lightsout: finished.
# mnist_puzzle mandrill_puzzle: finished.
# random_mnist_puzzle: not finished yet.

for type in fc2 ; do
    for task in hanoi mnist_puzzle mandrill_puzzle lenna_puzzle ; do
        for mode in learn ; do
            ./strips.py $type $task $mode
            samples/sync.sh
        done
    done
done

#  fcg2 1000
#  convg2 convg conv2 conv fc2 fc
# for type in convg ; do
#     for task in random_mnist_puzzle ; do
#         for mode in learn ; do
#             ./strips.py $type $task $mode
#             samples/sync.sh
#         done
#     done
# done
