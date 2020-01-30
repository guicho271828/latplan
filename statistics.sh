#!/bin/bash -x


dir=${1:-problem-instances}
allow_invalid=${2:false}

if ! $allow_invalid
then
    filter="map(select(.valid)) |"
else
    filter=""
fi

fn (){
    find $dir -name "*.json" -exec cat {} \; | jq -s --compact-output --monochrome-output --raw-output "$filter $*"
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

fn-search (){
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.search // 10000), (map(select(.heuristics==\"mands\")) | .[0].statistics.search // 10000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.search // 10000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}

fn-initialization (){
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.initialization // 10000), (map(select(.heuristics==\"mands\")) | .[0].statistics.initialization // 10000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.initialization // 10000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}

fn-total (){
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.total // 10000), (map(select(.heuristics==\"mands\")) | .[0].statistics.total // 10000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.total // 10000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}

fn-total2 (){
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.search+.[0].statistics.initialization // 10000), (map(select(.heuristics==\"mands\")) | .[0].statistics.search+.[0].statistics.initialization // 10000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.search+.[0].statistics.initialization // 10000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}

fn-expanded (){
    
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.expanded // 100000000), (map(select(.heuristics==\"mands\")) | .[0].statistics.expanded // 100000000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.expanded // 100000000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}
fn-evaluated (){
    
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.evaluated // 100000000), (map(select(.heuristics==\"mands\")) | .[0].statistics.evaluated // 100000000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.evaluated // 100000000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}
fn-generated (){
    
    fn "group_by([.noise, .domain, .problem, .domainfile[-1] ]) | map([ .[0].noise, .[0].domain, .[0].problem, .[0].domainfile[-1], (map(select(.heuristics==\"blind\")) | .[0].statistics.generated // 100000000), (map(select(.heuristics==\"mands\")) | .[0].statistics.generated // 100000000), (map(select(.heuristics==\"lmcut\")) | .[0].statistics.generated // 100000000) ] )" \
        | jq -r -c '.[]' | sed 's/\(\[\|\]\)//g'
}

fn-expanded       > $dir-expanded.csv
fn-generated      > $dir-generated.csv
fn-evaluated      > $dir-evaluated.csv
fn-search         > $dir-search.csv
fn-initialization > $dir-initialization.csv
fn-total          > $dir-total.csv
fn-total2         > $dir-total2.csv
