#!/bin/bash

trap exit SIGINT

d=$(dirname $(readlink -ef $0))

(
    cd $d

    $d/generate.py 7 100 puzzle mandrill 3 3
    $d/generate.py 7 100 puzzle spider   3 3
    $d/generate.py 7 100 puzzle mnist    3 3
    
    $d/generate.py 7 100 hanoi 7 4
    $d/generate.py 7 100 lightsout digital 4
    $d/generate.py 7 100 lightsout twisted 4
) &


(
    echo $d/../$(basename $d)-longest
    mkdir $d/../$(basename $d)-longest
    cd $d/../$(basename $d)-longest

    $d/generate.py 7 100 puzzle_longest mandrill 3 3
    $d/generate.py 7 100 puzzle_longest spider   3 3
    $d/generate.py 7 100 puzzle_longest mnist    3 3
) &

(
    echo $d/../$(basename $d)-gaussian
    mkdir $d/../$(basename $d)-gaussian
    cd $d/../$(basename $d)-gaussian

    $d/generate.py 7 100 noise gaussian puzzle mandrill 3 3
    $d/generate.py 7 100 noise gaussian puzzle spider   3 3
    $d/generate.py 7 100 noise gaussian puzzle mnist    3 3
    
    $d/generate.py 7 100 noise gaussian hanoi 7 4
    $d/generate.py 7 100 noise gaussian lightsout digital 4
    $d/generate.py 7 100 noise gaussian lightsout twisted 4
) &

(
    echo $d/../$(basename $d)-salt
    mkdir $d/../$(basename $d)-salt
    cd $d/../$(basename $d)-salt

    $d/generate.py 7 100 noise salt puzzle mandrill 3 3
    $d/generate.py 7 100 noise salt puzzle spider   3 3
    $d/generate.py 7 100 noise salt puzzle mnist    3 3

    $d/generate.py 7 100 noise salt hanoi 7 4
    $d/generate.py 7 100 noise salt lightsout digital 4
    $d/generate.py 7 100 noise salt lightsout twisted 4
) &

(
    echo $d/../$(basename $d)-pepper
    mkdir $d/../$(basename $d)-pepper
    cd $d/../$(basename $d)-pepper

    $d/generate.py 7 100 noise pepper puzzle mandrill 3 3
    $d/generate.py 7 100 noise pepper puzzle spider   3 3
    $d/generate.py 7 100 noise pepper puzzle mnist    3 3

    $d/generate.py 7 100 noise pepper hanoi 7 4
    $d/generate.py 7 100 noise pepper lightsout digital 4
    $d/generate.py 7 100 noise pepper lightsout twisted 4
) &

wait
echo "done!"
