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
from lspi import FastLSPI
from lspi import QR_LSPI
from td import Sarsa
import cPickle as pickle

run_chainwalk = False
run_gridworld = False
rbf_test = False
run_lspi = False
test_walls = True

if test_walls:
    # pdb.set_trace()
    gw = GridworldGui(nrows=5,ncols=5,endstates= [0], walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)])
    #gw = GridworldGui(nrows=5,ncols=5,endstates= [0], actions=[0,2], walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)])
    #t = gw.trace(1000)
    #pickle.dump(t,open("trace.pck","w"),pickle.HIGHEST_PROTOCOL)
    t = pickle.load(open("trace.pck"))
    policy0 = np.zeros(gw.nfeatures())

    w0, weights0 = LSPI(t, 0.001, gw, policy0, maxiter=100, method="sparse")    
    #w1, weights1 = QR_LSPI(t, 0.001, gw, policy0, maxiter=100, debug = False)

    w = w0
    pi = [gw.linear_policy(w,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)
    gw.background()
    # gw.redraw()        b = b + (features * r).T.toarray()[0]

    gw.mainloop()

if run_lspi:
    gw = GridworldGui(nrows = 9, ncols = 9, endstates = [0], walls = [])
    t = gw.trace(100000, show = False)
    policy0 = np.zeros(gw.nfeatures())

    pdb.set_trace()
        
    #w1, weights1 = FastLSPI(t, 0.01, gw, policy0, debug = False, timer = True)

    #w2, weights2 = LSPI(t, 0.001, gw, policy0, debug = False, timer = True)

    learner = Sarsa(8, 81, 0.5, 0.9, 0.9, 0.1)
    learner.learn(1000, gw, verbose=True)
    pi = [learner.best(s) for s in range(gw.nstates)]
    gw.set_arrows(pi)
    gw.background()
    gw.redraw()


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


