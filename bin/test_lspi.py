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
from gridworld.gridworld8 import wall_pattern
from gridworld.gridworldgui import GridworldGui, RBFGridworldGui
from lspi import LSTDQ
from lspi import LSPI
from lspi import FastLSTDQ
from lspi import OptLSTDQ
from lspi import LSPIRmax
from td import Sarsa
import cPickle as pickle

# Choose what tests to run.
test_rbf = False
test_scale= False
test_chainwalk = False
test_sarsa = False
test_lspi = False
test_walls = False
test_pca = False
test_rmax = True

if test_rmax:
    gw = GridworldGui(nrows = 5, ncols = 5, endstates = [0], walls = [])
    
    try:
        raise ValueError # for new trace
        t = pickle.load(open("rmax_trace.pck"))
    except:
        t = gw.trace(100, show = True)
        pickle.dump(t, open("rmax_trace.pck","w"), pickle.HIGHEST_PROTOCOL)

    policy0 = np.zeros(gw.nfeatures())
    # TODO - The tolerances for lsqr need to be related to the tolerances for the policy. Otherwise the number of iterations will be far larger than needed.
    w0, weights0 = LSPIRmax(t, 0.0003, gw, policy0, maxiter = 100, show = True, resample_epsilon = 0.1, rmax = 10)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)
    gw.background()
    gw.mainloop()

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
    w0, weights0 = LSPI(t, 0.00001, gw, policy0, maxiter=100, method="sparse", testing=False, debug = False, timer = True, show=True)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

if test_sarsa:
    gw = GridworldGui(nrows = 9, ncols = 9, endstates = [0], walls = [])
    learner = Sarsa(8, 81, 0.5, 0.9, 0.9, 0.1)
    learner.learn(10000, gw, verbose=True)
    pi = [learner.best(s) for s in range(gw.nstates)]
    gw.set_arrows(pi)
    gw.background()
    gw.mainloop()

if test_chainwalk:
    cw = Chainwalk()
    t = cw.trace(1000)
    policy0 = np.zeros(cw.nfeatures())
    print LSTDQ(t, cw, policy0)

if test_scale:
    gw = GridworldGui(nrows=64,ncols=64, size=8, endstates = [0], walls=[])
    try:
        t = pickle.load(open("scale_trace.pck"))
    except:
        t = gw.trace(100000)#, show = False)
        pickle.dump(t,open("scale_trace.pck","w"),pickle.HIGHEST_PROTOCOL)

    policy0 = np.zeros(gw.nfeatures())
    #w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=1, method="alt", debug = False, timer = True, show=False, format="csr")
    w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="parallel", debug = False, timer = True, show=True, format="csr",ncpus=6)
    #w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="sparse", debug = False, timer = True, show=True, format="csr")
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

if test_rbf:
    walls = wall_pattern(9,9)
    gw = RBFGridworldGui(nrows = 9, ncols = 9, walls = walls, endstates = [0], size=16, nrbf=15)
    # gw = Gridworld2(nrows = 9, ncols = 9, endstates = [0], walls = [], nrbf=15)
    t = gw.trace(1000)
    policy0 = np.zeros(gw.nfeatures())
    w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="sparse", debug = False, timer = True, show=True, format="csr",ncpus=6)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

if test_pca:
    endstates = [32, 2016, 1024, 1040, 1056, 1072]
    gw = GridworldGui(nrows=32,ncols=64,endstates=endstates,walls=[])
    try:
        t = pickle.load(open("pca_trace.pck"))
    except:
        t = gw.trace(1000)
        pickle.dump(t,open("pca_trace.pck","w"), pickle.HIGHEST_PROTOCOL)


    policy0 = np.zeros(gw.nfeatures())
    w0, weights0 = LSPIRmax(t, 0.003, gw, policy0, maxiter=100)
    #w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="parallel", debug = False, timer = True, show=True, format="csr",ncpus=6)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()
    

