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
