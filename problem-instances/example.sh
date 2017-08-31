#!/bin/bash

trap exit SIGINT

{
    cd $(dirname $0)

    ./generate.py 7 100 puzzle mandrill 3 3
    ./generate.py 7 100 puzzle spider   3 3
    ./generate.py 7 100 puzzle mnist    3 3

    ./generate.py 7 100 hanoi 7 4
    ./generate.py 7 100 lightsout digital 4
    ./generate.py 7 100 lightsout twisted 4
}


{
    mkdir $(dirname $0)/../$(basename $(dirname $0))-longest
    cd $(dirname $0)/../$(basename $(dirname $0))-longest

    ./generate.py 7 100 puzzle_longest mandrill 3 3
    ./generate.py 7 100 puzzle_longest spider   3 3
    ./generate.py 7 100 puzzle_longest mnist    3 3
}

{
    mkdir $(dirname $0)/../$(basename $(dirname $0))-gaussian
    cd $(dirname $0)/../$(basename $(dirname $0))-gaussian

    ./generate.py 7 100 noise gaussian puzzle mandrill 3 3
    ./generate.py 7 100 noise gaussian puzzle spider   3 3
    ./generate.py 7 100 noise gaussian puzzle mnist    3 3

    ./generate.py 7 100 noise gaussian hanoi 7 4
    ./generate.py 7 100 noise gaussian lightsout digital 4
    ./generate.py 7 100 noise gaussian lightsout twisted 4
}

{
    mkdir $(dirname $0)/../$(basename $(dirname $0))-salt
    cd $(dirname $0)/../$(basename $(dirname $0))-salt

    ./generate.py 7 100 noise salt puzzle mandrill 3 3
    ./generate.py 7 100 noise salt puzzle spider   3 3
    ./generate.py 7 100 noise salt puzzle mnist    3 3

    ./generate.py 7 100 noise salt hanoi 7 4
    ./generate.py 7 100 noise salt lightsout digital 4
    ./generate.py 7 100 noise salt lightsout twisted 4
}

{
    mkdir $(dirname $0)/../$(basename $(dirname $0))-pepper
    cd $(dirname $0)/../$(basename $(dirname $0))-pepper

    ./generate.py 7 100 noise pepper puzzle mandrill 3 3
    ./generate.py 7 100 noise pepper puzzle spider   3 3
    ./generate.py 7 100 noise pepper puzzle mnist    3 3

    ./generate.py 7 100 noise pepper hanoi 7 4
    ./generate.py 7 100 noise pepper lightsout digital 4
    ./generate.py 7 100 noise pepper lightsout twisted 4
}

