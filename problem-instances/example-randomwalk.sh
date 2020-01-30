#!/bin/bash

parallel --files <<EOF

# ./generate-randomwalk.py 7 5 puzzle_longest mandrill 3 3
# ./generate-randomwalk.py 7 5 puzzle_longest spider   3 3
# ./generate-randomwalk.py 7 5 puzzle_longest mnist    3 3

./generate-randomwalk.py 7 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mandrill 3 3
./generate-randomwalk.py 7 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle spider   3 3
./generate-randomwalk.py 7 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mnist    3 3
./generate-randomwalk.py 7 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout digital 4
./generate-randomwalk.py 7 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout twisted 4
./generate-randomwalk.py 7 5 noise saltpepper 0.06 noise gaussian 0.3 hanoi 4 8

./generate-randomwalk.py 14 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mandrill 3 3
./generate-randomwalk.py 14 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle spider   3 3
./generate-randomwalk.py 14 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mnist    3 3
./generate-randomwalk.py 14 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout digital 4
./generate-randomwalk.py 14 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout twisted 4
./generate-randomwalk.py 14 5 noise saltpepper 0.06 noise gaussian 0.3 hanoi 4 8

./generate-randomwalk.py 21 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mandrill 3 3
./generate-randomwalk.py 21 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle spider   3 3
./generate-randomwalk.py 21 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mnist    3 3
./generate-randomwalk.py 21 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout digital 4
./generate-randomwalk.py 21 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout twisted 4
./generate-randomwalk.py 21 5 noise saltpepper 0.06 noise gaussian 0.3 hanoi 4 8

./generate-randomwalk.py 28 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mandrill 3 3
./generate-randomwalk.py 28 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle spider   3 3
./generate-randomwalk.py 28 5 noise saltpepper 0.06 noise gaussian 0.3 puzzle mnist    3 3
./generate-randomwalk.py 28 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout digital 4
./generate-randomwalk.py 28 5 noise saltpepper 0.06 noise gaussian 0.3 lightsout twisted 4
./generate-randomwalk.py 28 5 noise saltpepper 0.06 noise gaussian 0.3 hanoi 4 8

EOF

echo "done!"
