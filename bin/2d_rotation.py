#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: 2D_ROTATION.PY
Date: Wednesday, March 20 2013
Description: 2d rotation test code. Taken from "test_realpca" section of test_lspi.py.

NOTE: I seem to have created trace_analysis.py to handle this.
"""
import pdb
import os
import cPickle as pickle
import numpy as np
import functools
from gridworld.gridworldgui import RBFObserverGridworldGui
from gridworld.gridworld8 import RBFObserverGridworld
from lspi import LSPI

def modify_endstates(t,old_endstates, new_endstates):
    new_trace = []
    for r in t:
        new = list(r)
        if r[3] in old_endstates:
            new[2] = 0.0
        if r[3] in new_endstates:
            new[2] = 1000.0
        new_trace.append(new)
    return new_trace


if __name__ == '__main__':

    workspace = "{0}/wrk/lspi/bin".format(os.environ['HOME'])

    if True:

        endstates = [272] # [16]
        
        ogw = RBFObserverGridworldGui("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None, nrbf=80)

        # ogw = RBFObserverGridworld("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None, nrbf=80)


        t = pickle.load(open(workspace + "/traces/complete_trace.pck"))

    def modify_endstates(t,old_endstates, new_endstates):
        new_trace = []
        for r in t:
            new = list(r)
            if r[3] in old_endstates:
                new[2] = 0.0
            if r[3] in new_endstates:
                new[2] = 1000.0
            new_trace.append(new)
        return new_trace
        
        
    old_endstates = [16,256,264,272,280,496]
    t = modify_endstates(t,old_endstates,endstates)


    policy0 = np.zeros(ogw.nfeatures())
    w0, weights0 = LSPI(t, 0.005, ogw, policy0, maxiter=50, method="sparse", show=True, ncpus=6)
    

    policy = functools.partial(ogw.linear_policy,w0)

    pi = [ogw.linear_policy(w0,s) for s in range(ogw.nstates)]
    ogw.set_arrows(pi)    


    result = ogw.test(policy)
    sucesses = []
    failures = []
    for (i,r) in result.items():
        if r:
            sucesses.append(i)
        else:
            failures.append(i)

    # print sucesses
    # print failures

    # s = np.zeros(ogw.nstates)
    # for (l,i) in result.items():
    #     s[l] = i
    # ogw.set_heatmap(s) 

    ogw.mainloop()

    if False:
        pdb.set_trace()  # enter debugger
        # endstates = [272]
        # ogw = RBFObserverGridworldGui("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None, nrbf=80)
        # t = pickle.load(open(workspace + "/traces/complete_trace.pck"))

        # (w0, weights) = pickle.load(open("{0}/weights/rbf_obs_weights.pck".format(workspace))


        # traces = ogw.evaluate_func_policy(ogw.perfect_policy)
        # pickle.dump(traces, open("perf_pca_traces_center.pck","w"),pickle.HIGHEST_PROTOCOL)
        traces = pickle.load(open("perf_pca_traces_center.pck"))
        from dtw import edit_distance_vc

        ematrix = np.zeros((512,512))
        for (i,t) in enumerate(traces):
            for (j,s) in enumerate(traces):
                    ematrix[i,j] = edit_distance_vc([e[1] for e in t], [l[1] for l in s], (1.0, 1.0, 2.0))

        # print ematrix
        pickle.dump(ematrix, open("ematrix.pck","w"), pickle.HIGHEST_PROTOCOL)
        ematrix = pickle.load(open('ematrix.pck'))
        from mds import mds
        pdb.set_trace()
        y,s = mds(ematrix)
        #from utils import scatter
        #scatter(y[:,0],y[:,1])
        import pylab
        pylab.plot(y[:,0],y[:,1])
        pylab.savefig('test3.png')


        # ogw.background()
        # ogw.mainloop()
