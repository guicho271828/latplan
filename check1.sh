#!/bin/bash


dir=${1:-problem-instances}

fn (){
    find $dir -name "*.json" -exec cat {} \; | jq -s --compact-output --monochrome-output --raw-output "$*"
}

# "network":network_dir,
# "problem":os.path.normpath(problem_dir).split("/")[-1],
# "domain" :os.path.normpath(problem_dir).split("/")[-2],
# "noise"  :os.path.normpath(problem_dir).split("/")[-3],
# "times":times,
# "heuristics":heuristics,
# "domainfile":domainfile.split("/"),
# "problemfile":problemfile,
# "planfile":planfile,
# "tracefile":tracefile,
# "csvfile":csvfile,
# "pngfile":pngfile,
# "jsonfile":jsonfile,
# "parameters":sae.parameters,
# "cost":len(plan),
# "valid":valid,

# keys=".heuristics"
# keys=".domain .heuristics"
# keys=".domain .domainfile[-1] .heuristics"
# keys=".domainfile[-1] .domain .heuristics"

fn2(){
    keys="$*"
    keys1=""
    keys1_first=true
    for key in $keys
    do
        if $keys1_first
        then
            keys1="$key"
            keys1_first=false
        else
            keys1="${keys1}, $key"
        fi
    done

    keys2=""
    keys2_first=true
    for key in $keys
    do
        if $keys2_first
        then
            keys2="(.[0] | $key)"
            keys2_first=false
        else
            keys2="${keys2}, (.[0] | $key)"
        fi
    done

    # echo $keys1
    # echo $keys2
    # ((map(select(.valid)) | length)/(map(select(.found)) | length))
    # first filter: ood-json gets in the way
    fn "map(select(.problem)) | group_by([$keys1]) | map([ $keys2, (map(select(.valid and ((.problem[0:3] | tonumber)==.statistics.cost))) | length), (map(select(.valid)) | length), (map(select(.found)) | (map(.plan_count)|add)//length), length ] | join(\" \"))" \
        | jq -r '.[]' | column -t
}

fn2 .domain "(.domainfile[-2]+.aae+.heuristics)"      | tee $1-coverage.csv
