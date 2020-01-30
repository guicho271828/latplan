#!/bin/bash

parallel --files <<EOF

./generate-dijkstra.py 7 30 puzzle mandrill 3 3
./generate-dijkstra.py 7 30 puzzle spider   3 3
./generate-dijkstra.py 7 30 puzzle mnist    3 3
./generate-dijkstra.py 7 30 lightsout digital 4
./generate-dijkstra.py 7 30 lightsout twisted 4

EOF

echo "done!"
