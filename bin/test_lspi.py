#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_LSPI.PY
Date: Wednesday, July 25 2012
Description: Test code for LSPI.
"""


import numpy as np
from gridworld.chainwalk import Chainwalk
from gridworld.gridworld8 import SparseGridworld8 as Gridworld
from gridworld.gridworld8 import SparseRBFGridworld8 as Gridworld2
from lspi import LSTDQ
from lspi import LSPI
from lspi import FastLSTDQ

run_chainwalk = False
run_gridworld = True
rbf_test = False

if rbf_test:
    gw2 = Gridworld2(nrows = 9, ncols = 9, endstates = [0], walls = [], nrbf=15)
    s = gw2.trace(10000)
    policy2 = np.zeros(gw2.nfeatures())

if run_gridworld:
    gw = Gridworld(nrows = 9, ncols = 9, endstates = [0], walls = [])
    t = gw.trace(10000)
    policy0 = np.zeros(gw.nfeatures())

    w1 = FastLSTDQ(t, gw, policy0, debug = False, timer=True)
    w2 = LSTDQ(t, gw, policy0, debug = False, timer=True)

if run_chainwalk:
    cw = Chainwalk()
    t = cw.trace(1000)
    policy0 = np.zeros(cw.nfeatures)
    print LSTDQ(t, cw, policy0)


