#!/bin/bash -x

set -e

trap exit SIGINT

ulimit -v 16000000000

dir=$(dirname $(dirname $(readlink -ef $0)))
proj=$(date +%Y%m%d%H%M)
# Example command for lsf job scheduler
common="--dry-run --keep-order -j 1 jbsub -mem 64g -cores 1+1 -queue x86_12h -proj $proj PYTHONPATH=$dir:$PYTHONPATH PYTHONUNBUFFERED=1"

class=ZeroSuppressConvolutionalGumbelAE      # Zero-suppressed SAE, inverted loss (pointy loss) for VAE
class=NGZeroSuppressConvolutionalGumbelAE    # NG-SAE   (i.e. the "correct" VAE in the paper)
class=NoKLZeroSuppressConvolutionalGumbelAE  # NoKL-SAE (i.e. without variational loss)
class=DetZeroSuppressConvolutionalGumbelAE   # Det-SAE  (i.e. without Gumbel noise, without variational loss)
# 
# Note: ZSAE with alpha=0.0 is same as the original SAE.
#                                                                                                           N             examples  alpha
parallel $common ./strips.py learn_plot_dump puzzle    ${class} mandrill 3 3 ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump puzzle    ${class} mnist 3 3    ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump puzzle    ${class} spider 3 3   ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump lightsout ${class} digital 4    ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./strips.py learn_plot_dump lightsout ${class} twisted 4    ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7

parallel $common ./state_discriminator3.py samples/lightsout_${class}_digital_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/lightsout_${class}_twisted_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/puzzle_${class}_mandrill_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/puzzle_${class}_mnist_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./state_discriminator3.py samples/puzzle_${class}_spider_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7

parallel $common ./action_autoencoder.py samples/lightsout_${class}_digital_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/lightsout_${class}_twisted_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/puzzle_${class}_mandrill_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/puzzle_${class}_mnist_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_autoencoder.py samples/puzzle_${class}_spider_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7

parallel $common ./action_discriminator.py samples/lightsout_${class}_digital_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/lightsout_${class}_twisted_4_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/puzzle_${class}_mandrill_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/puzzle_${class}_mnist_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
parallel $common ./action_discriminator.py samples/puzzle_${class}_spider_3_3_{1}_{2}_{3} learn_test ::: 36 64 100 ::: 10000 ::: 0.0 0.2 0.5 0.7
