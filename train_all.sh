#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
# Example command for lsf job scheduler
common="--dry-run --keep-order -j 1 jbsub -mem 64g -cores 1+1 -queue x86_12h -proj $proj PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1"

# Note: ZSAE with alpha=0.0 is same as normal SAE.
# Note: you can train NG-SAE (i.e. the "correct" VAE in the paper) by NGZeroSuppressConvolutionalGumbelAE
#                                                                                                           N             examples  alpha
parallel $common ./strips.py learn_plot_dump puzzle    ZeroSuppressConvolutionalGumbelAE mandrill 3 3 ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump puzzle    ZeroSuppressConvolutionalGumbelAE mnist 3 3    ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump puzzle    ZeroSuppressConvolutionalGumbelAE spider 3 3   ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump lightsout ZeroSuppressConvolutionalGumbelAE digital 4    ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump lightsout ZeroSuppressConvolutionalGumbelAE twisted 4    ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7

parallel $common ./state_discriminator3.py samples/lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/lightsout_ZeroSuppressConvolutionalGumbelAE_twisted_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_mandrill_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_mnist_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_spider_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7

parallel $common ./action_autoencoder.py samples/lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/lightsout_ZeroSuppressConvolutionalGumbelAE_twisted_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_mandrill_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_mnist_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_spider_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7

parallel $common ./action_discriminator.py samples/lightsout_ZeroSuppressConvolutionalGumbelAE_digital_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/lightsout_ZeroSuppressConvolutionalGumbelAE_twisted_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_mandrill_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_mnist_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/puzzle_ZeroSuppressConvolutionalGumbelAE_spider_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
