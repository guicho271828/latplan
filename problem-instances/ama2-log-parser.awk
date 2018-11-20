#!/usr/bin/awk -f

# usage:
# find -name "*.log" | parallel ./ama2-log-parser.awk > ama2-search-statistics.csv

# notice the ./
# ./vanilla/latplan.puzzles.lightsout_digital/007-000-000/lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_36_20000_0.0_False_Astar.log
# ./vanilla/latplan.puzzles.puzzle_mandrill/007-000-000/puzzle_ZeroSuppressConvolutionalGumbelAE_mandrill_3_3_36_20000_0.0_False_Astar.log


# then run
# scp ccc016:repos/latplan-zsae-icaps/latplan/noise-0.6-0.12-ama2/ama2-search-statistics.csv .

BEGIN {
    expansion = 0
    evaluation = 0
    generation = 0
    time = 0
}

/expanded/{ expansion   = $2 }
/generated/{ evaluation = $2 }
/generated_including_duplicate/{ generation = $2 }

/time/{ if (time==0) {time = $2} }


END {

    # file:
    # lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_36_20000_0.0_False_Astar.log
    # validity file
    # lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_36_20000_0.0_False_Astar_path_0.valid

    valid = gensub(/\.log/, "_path_0.valid", "g", FILENAME)
    solution = system("! test -e "valid)

    
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
    algo    = gensub(/\.log/, "", "g", tmp[offset+5])

    print noise","domain","step","problem","network","N","samples","zerosup","argmax","algo","solution","expansion","evaluation","generation","time","FILENAME
}

