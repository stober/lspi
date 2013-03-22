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
import pylab
from mds import mds
from isomap import isomap
from dtw import edit_distance_vc
from utils import inscatter, scatter
from procrustes import procrustes

def modify_endstates(t,old_endstates, new_endstates, action_costs = False):
    new_trace = []
    for r in t:
        new = list(r)
        if r[3] in old_endstates:
            new[2] = 0.0
        if r[3] in new_endstates:
            new[2] = 1000.0
        if action_costs:
            if r[1] in (0,2,4,6):
                new[2] += -1.0
            else:
                new[2] += -1.2 # diagonal actions are slightly more expensive


        new_trace.append(new)
    return new_trace

def callback(iters, current, env):
    print 'Callback: ', iters

    # get all traces
    env.updategui = False
    traces = env.evaluate_policy(current)
    env.updategui = True
    pickle.dump(traces, open('traces{0}.pck'.format(iters),'w'), pickle.HIGHEST_PROTOCOL)


    # measure task performance
    avg_reward = 0
    for t in traces:
        avg_reward += sum([i[2] for i in t])
    avg_reward = avg_reward / float(len(traces))
    print 'Avg reward: ', avg_reward


    # find current embedding
    ematrix = np.zeros((512,512))
    for (i,t) in enumerate(traces):
        for (j,s) in enumerate(traces):
            ematrix[i,j] = edit_distance_vc([e[1] for e in t], [l[1] for l in s], (1.0, 1.0, 1.2))
    pickle.dump(ematrix,open('ematrix{0}.pck'.format(iters), 'w'), pickle.HIGHEST_PROTOCOL)
    y,s,adj = isomap(ematrix)


    # plot stuff later because of pylab / pygame incompat on mac
    # save embedding image - multiple formats?
    #scatter(y[:,0],y[:,1], filename='scatter{0}.pdf'.format(iters))
    
    # save scree plot
    #plot(s[:10], filename='scree{0}.pdf'.format(iters))

    # procrustes error
    gt = env.coords_array()
    err = procrustes(gt, y)
    print "Procrustes ", err

    pickle.dump((iters, err, gt, avg_reward, current, y, s, adj), open('misc{0}.pck'.format(iters)), pickle.HIGHEST_PROTOCOL)

    env.save('iter{0}.png'.format(iters))


if __name__ == '__main__':

    workspace = "{0}/wrk/lspi/bin".format(os.environ['HOME'])

    if True:

        endstates = [272] # [16]
        
        ogw = RBFObserverGridworldGui("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None, nrbf=80)

        # ogw = RBFObserverGridworld("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = endstates, walls=None, nrbf=80)


        t = pickle.load(open(workspace + "/traces/complete_trace.pck"))

        
        
        old_endstates = [16,256,264,272,280,496]
        t = modify_endstates(t,old_endstates,endstates,action_costs=True)


        policy0 = np.zeros(ogw.nfeatures())
        w0, weights0 = LSPI(t, 0.005, ogw, policy0, maxiter=50, method='sparse', show=True, ncpus=6, callback=callback)


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

        traces = pickle.load(open("perf_pca_traces_center.pck"))

        ematrix = np.zeros((512,512))
        for (i,t) in enumerate(traces):
            for (j,s) in enumerate(traces):
                    ematrix[i,j] = edit_distance_vc([e[1] for e in t], [l[1] for l in s], (1.0, 1.0, 1.2))

        pickle.dump(ematrix, open("ematrix.pck","w"), pickle.HIGHEST_PROTOCOL)
        ematrix = pickle.load(open('ematrix.pck'))
        
        

        inscatter(isomap(ematrix))
