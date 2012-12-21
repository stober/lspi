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
from gridworld.gridworld8 import ObserverGridworld
from gridworld.gridworldgui import GridworldGui, RBFGridworldGui, ObserverGridworldGui, AliasGridworldGui, RBFObserverGridworldGui
from lspi import LSTDQ
from lspi import LSPI
from lspi import FastLSTDQ
from lspi import OptLSTDQ
from lspi import LSPIRmax
from td import Sarsa
import cPickle as pickle
import numpy.linalg as la
from utils import create_cluster_colors_rgb, find_duplicates

workspace = "/Users/stober/wrk/lspi/bin"

# Choose what tests to run.
test_rbf = False
test_comb = False
test_scale= False
test_chainwalk = False
test_sarsa = False
test_lspi = False
test_walls = False
test_fakepca = False
test_rmax = False
test_realpca = True #True
test_alias = False

def compare_all_phi(gw):
    data = []
    for s in gw.states:
        for a in gw.actions:
            data.append(gw.phi(s,a))
    print find_duplicates(data).next() # stop iteration?

if test_rmax:
    gw = GridworldGui(nrows = 5, ncols = 5, endstates = [0], walls = [])
    
    # try:
    #     raise ValueError # for new trace
    #     t = pickle.load(open("rmax_trace.pck"))
    # except:
    #     t = gw.trace(100, show = True)
    #     pickle.dump(t, open("rmax_trace.pck","w"), pickle.HIGHEST_PROTOCOL)

    policy0 = np.zeros(gw.nfeatures())
    t = []
    # TODO - The tolerances for lsqr need to be related to the tolerances for the policy. Otherwise the number of iterations will be far larger than needed.
    w0, weights0 = LSPIRmax(t, 0.003, gw, policy0, maxiter = 1000, show = True, resample_epsilon = 0.0, rmax = 1000)
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
    w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="parallel", debug = False, timer = True, show=True,ncpus=6)
    #w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="sparse", debug = False, timer = True, show=True, format="csr")
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

if test_rbf:
    walls = wall_pattern(9,9)
    gw = RBFGridworldGui(nrows = 9, ncols = 9, walls = walls, endstates = [0], nrbf=15)
    # gw = Gridworld2(nrows = 9, ncols = 9, endstates = [0], walls = [], nrbf=15)
    t = gw.trace(1000)
    policy0 = np.zeros(gw.nfeatures())
    w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="sparse", debug = False, timer = True, show=True, ncpus=6)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

if test_comb:
    walls = wall_pattern(9,9)
    # gw = GridworldGui(nrows = 9, ncols = 9, walls = walls, endstates = [0]) #, nrbf=15)
    # gw = Gridworld2(nrows = 9, ncols = 9, endstates = [0], walls = [], nrbf=15)
    # t = gw.trace(1000)        
    # policy0 = np.zeros(gw.nfeatures())
    # w0, weights0 = LSPIRmax(t, 0.005, gw, policy0, method = "dense", maxiter=1000, show=True, resample_epsilon = 0.1, rmax=1000)
    # w0 = pickle.load(open("weights.pck"))
    # pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    # gw.set_arrows(pi)    
    # gw.background()
    # pickle.dump(w0,open("weights.pck","w"), pickle.HIGHEST_PROTOCOL)
    #    gw.mainloop()
    # import pdb
    # pdb.set_trace()
    # traces = gw.evaluate_policy(w0)
    # pickle.dump(traces, open("traces.pck","w"),pickle.HIGHEST_PROTOCOL)
    traces = pickle.load(open("traces.pck"))
    from dtw import edit_distance

    ematrix = np.zeros((49,49))
    for (i,t) in enumerate(traces):
        for (j,s) in enumerate(traces):
                ematrix[i,j] = edit_distance([e[1] for e in t], [l[1] for l in s])

    print ematrix
    from mds import mds
    y,s = mds(ematrix)
    from utils import scatter
    scatter(y[:,0],y[:,1])
    import pylab
    pylab.show()

if test_fakepca:
    endstates = [32, 2016, 1024, 1040, 1056, 1072]
    gw = GridworldGui(nrows=32,ncols=64,endstates=endstates,walls=[])
    try:
        t = pickle.load(open("pca_trace.pck"))
    except:
        t = gw.trace(100000)
        pickle.dump(t,open("pca_trace.pck","w"), pickle.HIGHEST_PROTOCOL)


    policy0 = np.zeros(gw.nfeatures())
    w0, weights0 = LSPIRmax(t, 0.003, gw, policy0, maxiter = 100000, show = True, resample_epsilon = 0.1, rmax = 1000)
    # w0, weights0 = LSPIRmax(t, 0.003, gw, policy0, maxiter=100)
    # w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="parallel", debug = False, timer = True, show=True, format="csr",ncpus=6)
    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

if test_realpca:
    import pdb
    #pdb.set_trace()
    #endstates = [32, 2016, 1024, 1040, 1056, 1072]
    endstates = [16,256,264,272,280,496]
    #endstates = [0] # TODO find proper endstates
    # ogw = ObserverGridworldGui("/Users/stober/wrk/lspi/bin/16/5comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None)
    ogw = RBFObserverGridworldGui("/Users/stober/wrk/lspi/bin/16/5comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None)
    #ogw = ObserverGridworldGui("/Users/stober/wrk/lspi/bin/32/observations4.npy", "/Users/stober/wrk/lspi/bin/32/states.npy", endstates = endstates, walls=None)
    #ogw = GridworldGui(nrows=16,ncols=32,endstates = endstates, walls=[])
    try:
        #raise Exception # force a trace regeneration
        t = pickle.load(open(workspace + "/traces/real_pca_trace.pck"))
    except:
        t = ogw.trace(100000)
        pickle.dump(t, open(workspace + "/traces/real_pca_trace.pck","w"), pickle.HIGHEST_PROTOCOL)

    # rs = [r[2] for r in t]

    # s = np.zeros(ogw.nstates)
    # for r in t:
    #     s[r[0]] += 1

    # Sigh - not working - need to try tabular form (Tiling wrong?)

    #print t[:100]
    pdb.set_trace()
    policy0 = np.zeros(ogw.nfeatures())
    w0, weights0 = LSPI(t, 0.005, ogw, policy0, maxiter=50, method="dense", show=True, ncpus=6)
    #pi = [ogw.linear_policy(w0,s) for s in range(ogw.nstates)]
    #ogw.set_arrows(pi)    
    # print s[0],s[-1]
    # print np.max(s),np.min(s)

    # compare_all_phi(ogw)

    # obs = [ogw.observe(s)[1] for s in range(ogw.nstates)]
    # print obs
    # s = np.zeros(ogw.nstates)
    # s[:ogw.nstates/2] = 10.0
    # ogw.set_heatmap(obs)
    
    ogw.background()
    ogw.mainloop()

if test_alias:

    endstates = [32, 2016, 1024, 1040, 1056, 1072]

    dups = pickle.load(open("duplicates.pck"))

    import networkx as nx
    g = nx.Graph()
    for d,v in dups:
        g.add_edge(d,v)
    cc = nx.connected_components(g)
    aliases = {}
    for c in cc:
        aliases[min(c)] = set(c)
    print aliases


    #gw = AliasGridworldGui(nrows=32,ncols=64,endstates=endstates,walls=[],aliases=aliases)
    gw = GridworldGui(nrows=32,ncols=64,endstates=endstates,walls=[])
    try:
        t = pickle.load(open("pca_trace.pck"))
    except:
        t = gw.trace(100000)
        pickle.dump(t,open("pca_trace.pck","w"), pickle.HIGHEST_PROTOCOL)


    policy0 = np.zeros(gw.nfeatures())
    nclusters = len(aliases)
    colors = create_cluster_colors_rgb(nclusters,False)
    print colors

    colormap = {}
    for i,cc in enumerate(aliases.values()):
        for s in cc:
            colormap[s] = colors[i]

    w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="sparse", debug = False, timer = True, show=True, ncpus=6)
    #w0, weights0 = LSPIRmax(t, 0.003, gw, policy0, maxiter = 100000, show = True, resample_epsilon = 0.1, rmax = 1000)
    # w0, weights0 = LSPIRmax(t, 0.003, gw, policy0, maxiter=100)
    # w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=10, method="parallel", debug = False, timer = True, show=True, format="csr",ncpus=6)
    try:
        pickle.dump(w0,open("weights.pck","w"),pickle.HIGHEST_PROTOCOL)
    except:
        print "Save failed!"

    pi = [gw.linear_policy(w0,s) for s in range(gw.nstates)]
    gw.set_arrows(pi)    
    gw.background()
    gw.mainloop()

    # print max([len(cc) for cc in aliases.values()])
    # gw.set_colormap(colormap)
    # gw.background()
    # gw.mainloop()

    # print ogw.phi(0,0)
    # print ogw.phi(0,1)
    # print np.sum(rs)
    # import sys
    # sys.exit()

    # policy0 = np.zeros(ogw.nfeatures())
    # w0, weights0 = LSPI(t, 0.00001, ogw, policy0, maxiter=100, method="dense", debug = False, timer = True, show=True, ncpus=6)
    # pickle.dump(w0, open("weights.pck","w"), pickle.HIGHEST_PROTOCOL)    
    # pi = [ogw.linear_policy(w0,s) for s in range(ogw.nstates)]
    # ogw.set_arrows(pi)    
    # ogw.background()
    # ogw.mainloop()    
