#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: FIND_DUPLICATES.PY
Date: Thursday, November  1 2012
Description: Find duplicate images.
"""

import numpy as np

duplicates = []
imgs = np.load("observations.npy")
#imgs = np.load("imgs.npy")
for (i,p) in enumerate(imgs):
    for (j,q) in enumerate(imgs[i:]):
        if i != j+i and np.allclose(p,q):
            print i,j+i
            duplicates.append((i,j+i))

import cPickle as pickle
pickle.dump(duplicates, open("o_duplicates.pck","w"), pickle.HIGHEST_PROTOCOL)


