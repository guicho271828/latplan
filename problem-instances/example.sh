#!/bin/bash

parallel --files <<EOF

./generate.py 7 100 puzzle_longest mandrill 3 3
./generate.py 7 100 puzzle_longest spider   3 3
./generate.py 7 100 puzzle_longest mnist    3 3

./generate.py 7 100 puzzle mandrill 3 3
./generate.py 7 100 puzzle spider   3 3
./generate.py 7 100 puzzle mnist    3 3
./generate.py 7 100 lightsout digital 4
./generate.py 7 100 lightsout twisted 4
./generate.py 6 100 hanoi 4 3

./generate.py 7 100 noise gaussian puzzle mandrill 3 3
./generate.py 7 100 noise gaussian puzzle spider   3 3
./generate.py 7 100 noise gaussian puzzle mnist    3 3
./generate.py 7 100 noise gaussian lightsout digital 4
./generate.py 7 100 noise gaussian lightsout twisted 4
./generate.py 6 100 noise gaussian hanoi 4 3

./generate.py 7 100 noise salt puzzle mandrill 3 3
./generate.py 7 100 noise salt puzzle spider   3 3
./generate.py 7 100 noise salt puzzle mnist    3 3
./generate.py 7 100 noise salt lightsout digital 4
./generate.py 7 100 noise salt lightsout twisted 4
./generate.py 6 100 noise salt hanoi 4 3

./generate.py 7 100 noise pepper puzzle mandrill 3 3
./generate.py 7 100 noise pepper puzzle spider   3 3
./generate.py 7 100 noise pepper puzzle mnist    3 3
./generate.py 7 100 noise pepper lightsout digital 4
./generate.py 7 100 noise pepper lightsout twisted 4
./generate.py 6 100 noise pepper hanoi 4 3

EOF

echo "done!"
