#!/bin/env python3

from setuptools import setup, find_packages

setup(name='latplan',
      version='0.0.1',
      install_requires=[],
      entry_points = {'console_scripts': ["latplan = latplan:main.main"] },
      packages=find_packages(),
      include_package_data=True,
)
