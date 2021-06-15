#!/bin/bash


echo downloading the dataset...
curl -L https://github.com/guicho271828/latplan/releases/download/v5.0.0/datasets.tar -o datasets.tar
curl -L https://github.com/guicho271828/latplan/releases/download/v5.0.0/backup-propositional.tar.bz2 -o backup-propositional.tar.bz2

( cd latplan/puzzles/ ; tar xf ../../datasets.tar )
( cd problem-generators/ ; tar xf ../backup-propositional.tar.bz2 )
