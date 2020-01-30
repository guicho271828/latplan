#!/bin/bash

parallel --files <<EOF

./generate-korf.py 0 0 korf mandrill 4 4

EOF

echo "done!"
