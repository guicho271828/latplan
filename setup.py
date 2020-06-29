#!/bin/env python3

from setuptools import setup, find_packages

setup(name='latplan',
      version='0.0.1',
      install_requires=[
          'tensorflow==1.15.2',
          'keras==2.2.5',
          'h5py',
          'matplotlib',
          'progressbar2',
          'keras-adabound',
          'keras-rectified-adam',
          'timeout_decorator',
          'ansicolors',
          'scipy==1.4.1',
          'scikit-image',
          'imageio',
          'pillow',],
      packages=find_packages(),
      include_package_data=True,
)
