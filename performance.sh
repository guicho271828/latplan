#!/bin/bash

if [ -z "$*" ]
then
    echo "Usage: $0 [QUERY...]" >&2
    echo "Usage: $0 [DIRECTORIES...] -- [QUERY...]" >&2
    echo "Example:"             >&2
    echo "$0 .sae.MSE.vanilla.test .sae.inactive.both.test .sae.hamming.test .sae.parameters.zerosuppress .sae.parameters.locality .aae.parameters.lr  .ad.parameters.lr  .sd.parameters.lr " >&2
    exit 1
fi

export samples=samples
export SHELL=/bin/bash

sep (){
    f=$(mktemp)
    for arg in $@
    do
        if [[ "$arg" == "--" ]]
        then
            : remaining args are queries
            echo -n "$f "
            f=$(mktemp)
        else
            echo "$arg" >> $f
        fi
    done
    echo $f
}

sep $@ | {
    read first second
    if [[ -z $second ]]
    then
        directories="samples/*/"
        queries="$(cat $first)"
    else
        directories=$(cat $first)
        queries="$(cat $second)"
    fi
    export directories queries


    rm $first $second

    json (){
        local root=${1%%/}
        local filename=$2
        local sae=$(cat ${root}/$filename      2>/dev/null || echo "{}")
        local aae=$(cat ${root}/_aae/$filename 2>/dev/null || echo "{}")
        local sd=$(cat ${root}/_sd3/$filename  2>/dev/null || echo "{}")
        local ad=$(cat ${root}/_ad/$filename   2>/dev/null || echo "{}")
        local ad2=$(cat ${root}/_ad2/$filename   2>/dev/null || echo "{}")
        echo "{\"sae\": $sae, \"aae\": $aae, \"sd\": $sd, \"ad\": $ad, \"ad2\": $ad2 } "
    }

    export -f json

    row (){
        performance=$(json "$2" performance.json)
        aux=$(json "$2" aux.json)
        if (echo "$performance $aux" | jq -s ".[0] * .[1]" | jq -e "$1" &>/dev/null)
        then
            (echo "$performance $aux" | jq -s ".[0] * .[1]" | jq -e "$1")
        else
            echo nan
        fi
    }

    export -f row

    col (){
        local f=$(mktemp) 
        echo $1 > $f
        # for d in $directories
        # do
        #     row "$1" $d >> $f
        # done
        # parallel --keep-order "row '$1'" ::: $directories | ./colorize.ros -i >> $f
        parallel --keep-order "row '$1'" ::: $directories >> $f
        echo $f
    }

    export -f col

    header (){
        local f=$(mktemp)
        echo directory > $f
        # for dir in $directories
        # do
        #     echo $dir >> $f
        # done
        parallel --keep-order echo ::: $directories >> $f
        echo $f
    }

    export -f header

    # files=$(header ; for query in $queries ; do col "$query" ; done)
    files=$(header ; parallel --keep-order col ::: $queries)
    paste $files | column -t
    rm $files

}
