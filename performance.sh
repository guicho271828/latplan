#!/bin/bash

if [ -z "$*" ]
then
    echo "Usage: $0 [QUERY...]" >&2
    echo "Example:"             >&2
    echo "$0 .sae.MSE.vanilla.test .sae.inactive.both.test .sae.hamming.test .sae.parameters.zerosuppress .sae.parameters.locality .aae.parameters.lr  .ad.parameters.lr  .sd.parameters.lr " >&2
    exit 1
fi

export samples=samples
export SHELL=/bin/bash

json (){
    root=${1%%/}
    filename=$2
    sae=$(cat ${root}/$filename      2>/dev/null || echo "{}")
    aae=$(cat ${root}/_aae/$filename 2>/dev/null || echo "{}")
    sd=$(cat ${root}/_sd3/$filename  2>/dev/null || echo "{}")
    ad=$(cat ${root}/_ad/$filename   2>/dev/null || echo "{}")
    ad2=$(cat ${root}/_ad2/$filename   2>/dev/null || echo "{}")
    echo "{\"sae\": $sae, \"aae\": $aae, \"sd\": $sd, \"ad\": $ad, \"ad2\": $ad2 } "
}

export -f json

row (){
    if json $2 performance.json | jq -e $1 &>/dev/null
    then
        json $2 performance.json | jq -e $1
        exit
    fi
    if json $2 aux.json | jq -e $1 &>/dev/null
    then
        json $2 aux.json | jq -e $1
        exit
    fi
    echo nan
}

export -f row

col (){
    f=$(mktemp) 
    echo $1 > $f
    parallel --keep-order row $1 :::  $samples/*/ >> $f
    echo $f
}

export -f col

header (){
    f=$(mktemp)
    echo dir > $f
    ls -vd $samples/*/ >> $f
    echo $f
}

export -f header

files=$(header ; parallel --keep-order col ::: $@)
paste $files | column -t
rm $files

