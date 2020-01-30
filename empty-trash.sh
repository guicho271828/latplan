#!/bin/bash -x

garbage=${1:-garbage}

garbage=${garbage%%/}

mkdir garbage
d=$(mktemp -d)
trap "rmdir $d" EXIT

echo rsync -r --delete $d/ $garbage/

echo "Do you wish to run: rsync -r --delete $d/ $garbage/ ?"
select yn in "yes" "no"; do
    case $yn in
        yes ) rsync -r --delete $d/ $garbage/; break;;
        no ) exit;;
    esac
done


