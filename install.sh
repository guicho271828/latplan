#!/bin/bash
set -e

# errexit(){
#     echo $@ >&2
#     exit 1
# }

# if ! uname -a | grep Ubuntu ; then
#     echo "This is not ubuntu! Modify install.sh by yourself" >&2
# fi

{
    sudo apt install -y mercurial g++ cmake make python flex bison g++-multilib
    git submodule update --init --recursive
    cd downward ; ./build.py
}


{
    # https://github.com/roswell/roswell/wiki/1.-Installation
    sudo apt -y install git build-essential automake libcurl4-openssl-dev
    git clone -b release https://github.com/roswell/roswell.git
    cd roswell
    sh bootstrap
    ./configure
    make
    sudo make install
    ros setup
    make -j 1 -C lisp
}

{
    sudo apt -y install python3-pip python3-pil
    pip3 install --user \
         tensorflow tensorflow-gpu keras h5py matplotlib progressbar2 json \
         timeout_decorator ansicolors hashlib fcntl queue scipy scikit-image
    mkdir -p ~/.keras
    cp keras-tf.json ~/.keras/keras.json
}

    
