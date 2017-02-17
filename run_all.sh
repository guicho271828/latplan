#!/bin/bash

trap exit SIGINT

for type in fc2 ; do
    for task in hanoi4 hanoi10 mnist_puzzle digital_lightsout mnist_puzzle mandrill_puzzle ; do
        for mode in learn_dump ; do
            ./strips.py $type $task $mode
            samples/sync.sh
        done
    done
done

for type in fc ; do
    for task in digital_lightsout_skewed3 ; do
        for mode in learn_dump ; do
            ./strips.py $type $task $mode
            samples/sync.sh
        done
    done
done

