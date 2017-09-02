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

./generate.py 7 100 noise gaussian 0.3 puzzle mandrill 3 3
./generate.py 7 100 noise gaussian 0.3 puzzle spider   3 3
./generate.py 7 100 noise gaussian 0.3 puzzle mnist    3 3
./generate.py 7 100 noise gaussian 0.3 lightsout digital 4
./generate.py 7 100 noise gaussian 0.3 lightsout twisted 4
./generate.py 6 100 noise gaussian 0.3 hanoi 4 3

./generate.py 7 100 noise saltpepper 0.06 puzzle mandrill 3 3
./generate.py 7 100 noise saltpepper 0.06 puzzle spider   3 3
./generate.py 7 100 noise saltpepper 0.06 puzzle mnist    3 3
./generate.py 7 100 noise saltpepper 0.06 lightsout digital 4
./generate.py 7 100 noise saltpepper 0.06 lightsout twisted 4
./generate.py 6 100 noise saltpepper 0.06 hanoi 4 3

EOF

echo "done!"
