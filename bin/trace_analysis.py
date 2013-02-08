#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TRACE_ANALYSIS.PY
Date: Thursday, February  7 2013
Description: Check trace properties.
"""

import cPickle as pickle
import numpy as np
import pylab
import numpy.linalg as la
from mds import mds
from multiprocessing import Pool

if True:
    #ematrix = pickle.load(open("ematrix4.pck"))

    traces = pickle.load(open("perf_pca_traces_center.pck"))
    from dtw import dtw_distance

    # taken from gridworld8
    actions = [np.array((-1,0)),np.array((-1,-1)),
               np.array((0,-1)),np.array((1,-1)),
               np.array((1,0)), np.array((1,1)),
               np.array((0,1)), np.array((-1,1))]

    adistances = np.zeros((8,8))
    for (i,s) in enumerate(actions):
        for (j,t) in enumerate(actions):
            adistances[i,j] = la.norm(s - t)

    def cost_func(a,b):
        """
        Real costs for actions.
        """
        return adistances[a,b]

    def dtw_apply(i,j,t,s):
        return i,j,dtw_distance([e[1] for e in t], [l[1] for l in s], costf=cost_func)

    pool = Pool(6)
    results = []
    for (i,t) in enumerate(traces):
        for (j,s) in enumerate(traces):
            r = pool.apply_async(dtw_apply,(i,j,t,s))
            results.append(r)

    ematrix = np.zeros((512,512))     
    for r in results:
        (i,j,d) = r.get()
        ematrix[i,j] =  d
     
    #ematrix[i,j] = dtw_distance([e[1] for e in t], [l[1] for l in s], costf=cost_func)
    np.save('test_matrix.npy',ematrix)
    y,s = mds(ematrix)
    #pylab.scatter(y[:,0],y[:,1])
    #pylab.show()


if False:
    traces = pickle.load(open("perf_pca_traces.pck"))
    from dtw import edit_distance_vc

    ematrix = np.zeros((512,512))
    for (i,t) in enumerate(traces):
        for (j,s) in enumerate(traces):
            ematrix[i,j] = edit_distance_vc([e[1] for e in t], [l[1] for l in s], (1.0, 1.0, 1.2))

    np.save('ematrix5.npy',ematrix)

    y,s = mds(ematrix)
    pylab.scatter(y[:,0],y[:,1])
    pylab.show()

if False:
    comps = np.load("/Users/stober/wrk/lspi/bin/16/20comp.npy")
    pmatrix = np.zeros((512,512))

    for (i,t) in enumerate(comps):
        for (j,s) in enumerate(comps):
            pmatrix[i,j] = la.norm(t - s)

    y,s = mds(pmatrix)
    pylab.scatter(y[:,0],y[:,1])
    pylab.show()

