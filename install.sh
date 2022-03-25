#!/bin/bash

env=latplan

# execute it in a subshell so that set -e stops on error, but does not exit the parent shell.
(
    set -e

    (conda activate >/dev/null 2>/dev/null)  || {
        echo "This script must be sourced, not executed. Run it like: source $0"
        exit 1
    }

    conda env create -n $env -f environment.yml || {
        echo "installation failed; cleaning up"
        conda env remove -n $env
        exit 1
    }

    conda activate $env

    git submodule update --init --recursive

    ################################################################
    # reset my local variables so that they do not affect reproducibility

    echo "path variables that are set locally:"
    echo "-----------------"
    env | grep PATH | cut -d= -f1
    echo "-----------------"
    echo

    conda env config vars set LD_LIBRARY_PATH=
    conda env config vars set PYTHONPATH=
    conda env config vars set PKG_CONFIG_PATH=

    # these are required for building dependencies properly under conda
    conda env config vars set CPATH=${CONDA_PREFIX}/include:${CONDA_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/include:
    conda env config vars set ROSWELL_HOME=$(readlink -ef .roswell)

    # note: "variables" field in yaml file introduced in conda 4.9 does not work because it does not expand shell variables

    # these are required for the variables to take effect
    conda deactivate ; conda activate $env

    echo $CPATH ; echo $LD_LIBRARY_PATH

    conda env config vars list

    ################################################################
    # build fast downward

    (
        cd downward
        ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
    )

    ################################################################
    # build ARRIVAL validator through roswell
    (
        cd roswell
        sh bootstrap
        ./configure --prefix=${CONDA_PREFIX}
        make
        make install
        ros setup
        ros dynamic-space-size=8000 install numcl arrival eazy-gnuplot magicffi dataloader
    )

    make -j 1 -C lisp
    ./download-dataset.sh

) && {
    conda activate $env >/dev/null 2>/dev/null || echo "This script must be sourced, not executed. Run it like: source $0"
}

