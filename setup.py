#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: SETUP.PY
Date: Friday, November 11 2011
Description: Setup and install LSPI.
"""


from distutils.core import setup

setup(name='lspi',
      version='0.01',
      description='Least Squares Policy Iteration in Python',
      author="Jeremy Stober",
      author_email="stober@gmail.com",
      package_dir={"lspi" : "src"},
      packages=["lspi"]
      )
      
