#!/bin/bash -x

export T="1 2 5 10 20 40 80"
export D="4 7 12 25 50 100"

pre-acc-1 (){
    awk '{print $2}' accuracy/PU-BINARY-$1-$2.csv
}
pre-tpr-1 (){
    awk '{print $3}' accuracy/PU-BINARY-$1-$2.csv
}
pre-tnr-1 (){
    awk '{print $4}' accuracy/PU-BINARY-$1-$2.csv
}
pre-f-1 (){
    awk '{print $5}' accuracy/PU-BINARY-$1-$2.csv
}
eff-1 (){
    awk '{print $6}' accuracy/PU-BINARY-$1-$2.csv
}
filesize-1 (){
    du -h PDDL/dsama-PU-BINARY-$1-$2.pddl | awk '{print $1}'
}

pre-acc-2 (){
    awk '{print $2}' accuracy-with-successors/PU-BINARY-$1-$2.csv
}
pre-tpr-2 (){
    awk '{print $3}' accuracy-with-successors/PU-BINARY-$1-$2.csv
}
pre-tnr-2 (){
    awk '{print $4}' accuracy-with-successors/PU-BINARY-$1-$2.csv
}
pre-f-2 (){
    awk '{print $5}' accuracy-with-successors/PU-BINARY-$1-$2.csv
}
eff-2 (){
    awk '{print $6}' accuracy-with-successors/PU-BINARY-$1-$2.csv
}
filesize-2 (){
    du -h PDDL-WITH-SUCCESSORS/dsama-PU-BINARY-$1-$2.pddl | awk '{print $1}'
}

per-d (){
    tmp=$(mktemp)
    # trap "rm $tmp" RETURN
    $fn $1 $2 > $tmp
    echo $tmp
}

per-t (){
    files=$(parallel -j 1 --keep-order per-d $1 ::: $D)
    paste <(echo $1) $files
    rm $files
}

per-number (){
    export fn=$1
    echo 
    echo $fn
    (
        echo "T\\D $D"
        parallel -j 1 --keep-order per-t ::: $T
    ) | column -t
}

per-domain (){
    domain=$1
    cd $domain
    echo
    echo $(basename $domain)
    parallel -j 1 --keep-order per-number ::: {pre-acc,pre-f,pre-tpr,pre-tnr,eff,filesize}-{1,2}
}

export SHELL=/bin/bash
export -f per-domain per-number per-t per-d {pre-acc,pre-f,pre-tpr,pre-tnr,eff,filesize}-{1,2}

parallel --keep-order per-domain ::: samples/*
