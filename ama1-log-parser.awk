#!/usr/bin/awk -f

# usage:
# find problem-instances -name "*.sasp.gz.log" | parallel ./ama1-log-parser.awk > ama1-search-statistics.csv

# problem-instances/gaussian/latplan.puzzles.hanoi/007-000-000/hanoi_4_3_36_81_conv_all_actions.sasp.log
# problem-instances/gaussian/latplan.puzzles.lightsout_digital/007-000-000/lightsout_digital_4_36_20000_convz_all_actions.sasp.gz.log
# problem-instances/gaussian/latplan.puzzles.puzzle_spider/007-000-000/puzzle_spider_3_3_36_20000_conv_all_actions.sasp.log

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
    split(info[5],tmp,".")
    split(tmp[1],tmp2,"_")

    if (domain == "hanoi"){
        N       = tmp2[4]
        samples = tmp2[5]
        network = tmp2[6]
    }
    if (domain == "lightsout_digital" || domain == "lightsout_twisted"){
        N       = tmp2[4]
        samples = tmp2[5]
        network = tmp2[6]
    }
    if (domain == "puzzle_mnist" || domain == "puzzle_mandrill" || domain == "puzzle_spider"){
        N       = tmp2[5]
        samples = tmp2[6]
        network = tmp2[7]
    }

    print domain","step","noise","problem","N","samples","network","solution","expansion","evaluation","generation","time","FILENAME
}

