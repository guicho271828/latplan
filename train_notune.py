#!/usr/bin/env python3

import latplan.main
from train_common import parameters

parameters.update({
    'N'              :[50,100,300], # latent space size
    'zerosuppress'   :0.1,
    'beta_d'         :[ 1 ],
    'beta_z'         :[ 1 ],
})

if __name__ == '__main__':
    latplan.main.main(parameters)
