#!/bin/bash

parallel --files <<EOF

./generate-dijkstra.py 7  20 puzzle mandrill 4 4
./generate-dijkstra.py 14 20 puzzle mandrill 4 4

EOF

echo "done!"
