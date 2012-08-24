#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_LSPI.PY
Date: Wednesday, July 25 2012
Description: Test code for LSPI.
"""

import pdb
import sys
import numpy as np
from gridworld.chainwalk import Chainwalk
from gridworld.gridworld8 import SparseGridworld8 as Gridworld
from gridworld.gridworld8 import SparseRBFGridworld8 as Gridworld2
from gridworld.gridworldgui import GridworldGui
from lspi import LSTDQ
from lspi import LSPI
from lspi import FastLSTDQ
from lspi import OptLSTDQ
from td import Sarsa
import cPickle as pickle

# Choose what tests to run.
run_chainwalk = False
run_gridworld = False
rbf_test = False
test_lspi = True
test_walls = False

if test_walls:
    gw = GridworldGui(nrows=5,ncols=5,endstates= [0], walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)])
    try:
        t = pickle.load(open("walls_trace.pck"))
    except:
        t = gw.trace(1000, show=False)
        pickle.dump(t,open("walls_trace.pck","w"),pickle.HIGHEST_PROTOCOL)
    
    policy0 = np.zeros(gw.nfeatures())
    # TODO - The tolerances for lsqr need to be related to the tolerances for the policy. Otherwise the number of iterations will be far larger than needed.
    w0, weights0 = LSPI(t, 0.003, gw, policy0, maxiter=100, method="opt", show=True, debug=False)    
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)
    gw.background()
    gw.mainloop()

if test_lspi:
    gw = GridworldGui(nrows = 9, ncols = 9, endstates = [0], walls = [])
    try:
        t = pickle.load(open("lspi_trace.pck"))
    except:
        t = gw.trace(10000, show = False)
        pickle.dump(t,open("lspi_trace.pck","w"),pickle.HIGHEST_PROTOCOL)
    
    policy0 = np.zeros(gw.nfeatures())
    w0, weights0 = LSPI(t, 0.001, gw, policy0, maxiter=100, method="sparse", debug = False, timer = True, show=True)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

    # learner = Sarsa(8, 81, 0.5, 0.9, 0.9, 0.1)
    # learner.learn(10000, gw, verbose=True)
    # pi = [learner.best(s) for s in range(gw.nstates)]
    # gw.set_arrows(pi)
    

if rbf_test:
    gw2 = Gridworld2(nrows = 9, ncols = 9, endstates = [0], walls = [], nrbf=15)
    s = gw2.trace(10000)
    policy2 = np.zeros(gw2.nfeatures())

if run_gridworld:

    # stober@stobers-MacBook-Pro:~/Dropbox/workspace/lspi/bin$ python test_lspi.py # 9 x 9
    # # seconds:  15.244505167
    # # seconds:  3.86740398407
    # stober@stobers-MacBook-Pro:~/Dropbox/workspace/lspi/bin$ python test_lspi.py # 18 x 18
    # # seconds:  461.543802977
    # # seconds:  4.17448186874

    gw = Gridworld(nrows = 5, ncols = 5, endstates = [0], walls = [])
    t = gw.trace(10000)
    policy0 = np.zeros(gw.nfeatures())

    A1,b1,w1 = FastLSTDQ(t, gw, policy0, debug = False, timer=True)
    A2,b2,w2 = LSTDQ(t, gw, policy0, debug = False, timer=True)
    A3,b3,w3 = OptLSTDQ(t, gw, policy0, debug = False, timer=True)


    print "Fast LSTDQ Error:"
    print np.sum(np.abs(np.dot(A1.todense(),w1) - b1))

    print "Regular Error:"
    print np.sum(np.abs(np.dot(A2,w2) - b2))

    print "Opt Error:"
    print np.sum(np.abs((np.dot(A3,b3) - w3).todense()))

    pdb.set_trace()

    # >>> np.sum(np.abs(np.dot(A1.todense(),w1) - b1))
    # 0.041507042090663027
    # >>> np.sum(np.abs(np.dot(A2,w2) - b2))
    # 0.034436035156235029


if run_chainwalk:
    cw = Chainwalk()
    t = cw.trace(1000)
    policy0 = np.zeros(cw.nfeatures)
    print LSTDQ(t, cw, policy0)


