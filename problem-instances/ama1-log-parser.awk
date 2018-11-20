#!/usr/bin/awk -f

# usage:
# find -name "*.ama1.log" | parallel ./ama1-log-parser.awk > ama1-search-statistics.csv

# notice the ./
# ./vanilla/latplan.puzzles.lightsout_digital/007-000-000/lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_36_20000_0.0_False_blind.ama1.log
# ./vanilla/latplan.puzzles.puzzle_spider/007-000-000/puzzle_ZeroSuppressConvolutionalGumbelAE_spider_3_3_36_20000_0.0_False_blind.ama1.log
# scp ccc016:repos/latplan-zsae-icaps/latplan/noise-0.6-0.12-ama1/ama1-search-statistics.csv .

BEGIN {
    solution = 0
    expansion = 0
    evaluation = 0
    generation = 0
    time = 0
}

/Solution found!/{ solution = 1 }

/Expanded [0-9]* state\(s\)\./{ expansion   = $2 }
/Evaluated [0-9]* state\(s\)\./{ evaluation = $2 }
/Generated [0-9]* state\(s\)\./{ generation = $2 }

/Actual search time:/{ time = $4 }


END {
    split(FILENAME,info,"/")
    noise = info[2]
    split(info[3],tmp,".")
    domain = tmp[3]
    split(info[4],tmp,"-")
    step = tmp[1]
    problem = tmp[2]
    
    split(info[5],tmp,"_")
    metadomain = tmp[1]
    network = tmp[2]
    

    if (metadomain == "lightsout"){
        offset = 4
    }
    if (metadomain == "puzzle"){
        offset = 5
    }
    N       = tmp[offset+1]
    samples = tmp[offset+2]
    zerosup = tmp[offset+3]
    argmax  = tmp[offset+4]
    algo    = tmp[offset+5]

    print noise","domain","step","problem","network","N","samples","zerosup","argmax","algo","solution","expansion","evaluation","generation","time","FILENAME
}

