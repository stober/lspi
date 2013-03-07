#!/usr/bin/env python
'''
@author jstober
'''

import pdb
import sys
import numpy as np
import numpy.linalg as la
import functools
import os
from utils import scatter3d, create_norm_colors, save_many, scatter
from gridworld.gridworld8 import SparseGridworld8 as Gridworld
from gridworld.gridworld8 import SparseRBFGridworld8 as Gridworld2
from gridworld.gridworld8 import wall_pattern
from gridworld.gridworld8 import ObserverGridworld
from gridworld.gridworldgui import GridworldGui
from lspi import LSPI
from lspi import LSPIRmax
import cPickle as pickle
from sdp import write_spda_file, read_sol_file
import pylab
from isomap import shortest_paths, cluster_graph
import matplotlib as mpl
from mds import mds
from dtw import edit_distance

# ISOMAP is worse that MVU
# MVU is still bad (scree plot is ok) - still not sure why mvu doesn't
# use more dimensions for embedding?

# TODO: Color code and compare
# OPTIONAL: multiple comb sizes
# TODO: Finalize some viz.
# OPTIONAL: Procrustes analysis


def expand(N, k=None):
    X = np.zeros(N.shape)
    X[:, :] = np.inf
    X[N.nonzero()] = N[N.nonzero()]
    for i in range(X.shape[0]):
        X[i, i] = 0

    X = shortest_paths(X)
    if k:
        I = cluster_graph(X, fnc='k', size=k)
        J = np.zeros(N.shape)
        J[I] = X[I]
        return J
    else:
        return X

if __name__ == '__main__':

    walls = wall_pattern(9, 9)
    gw = Gridworld(nrows=9, ncols=9, walls=walls, endstates=[8])
    pts = np.array(gw.states.values())
    colors = create_norm_colors(pts)

    pylab.scatter(pts[:, 0], pts[:, 1], c=colors)
    save_many('exp_comb/gt')

    #
    # SE
    #

    # walls = wall_pattern(9, 9)
    # gw = GridworldGui(nrows=9, ncols=9, walls=walls, endstates=[0])  # if you want a gui
    t = gw.complete_trace()  # not a random sample - sample each transition once
    policy0 = np.zeros(gw.nfeatures())
    w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=100, method="sparse")

    # pi = [gw.linear_policy(w0, s) for s in range(gw.nstates)]
    # gw.set_arrows(pi)
    # gw.background()

    # dump weights in case we need them in the future
    pickle.dump(weights0, open("exp_comb/weights.pck", "w"), pickle.HIGHEST_PROTOCOL)

    # generate traces for sensorimotor embedding
    traces = gw.evaluate_policy(w0)
    pickle.dump(traces, open("exp_comb/comb_traces.pck", "w"), pickle.HIGHEST_PROTOCOL)

    ematrix = np.zeros((49, 49))
    for (i, t) in enumerate(traces):
        for (j, s) in enumerate(traces):
            ematrix[i, j] = edit_distance([e[1] for e in t], [l[1] for l in s])

    y, s = mds(ematrix)

    pylab.clf()
    pylab.scatter(y[:, 0], y[:, 1], c=colors)
    save_many('exp_comb/se')


    #
    # MVU
    #

    N = gw.neighbors([0, 2, 4, 6])

    # base colors on true points
    pts = np.array(gw.states.values())
    colors = create_norm_colors(pts)

    # N = expand(N, 20)
    sdp_filename = 'exp_comb/comb.sdp'
    sol_filename = 'exp_comb/comb.sol'

    x, y = np.nonzero(N)
    indx = zip(x, y)
    size = N.shape[0]
    m = len(indx) + 1
    nblocks = 1
    c = [0.0]
    for i, j in indx:
        c.append(1.0)

    write_spda_file(sdp_filename, m, nblocks, size, c, indx)
    os.system("/usr/local/bin/csdp {0} {1}".format(sdp_filename,
                                                       sol_filename))
    y, Z, X = read_sol_file(sol_filename, size=49)
    u, s, v = la.svd(X)
    Y = u * np.sqrt(s)

    pylab.clf()
    pylab.plot(s[:10])
    save_many('exp_comb/mvu_scree')


    pylab.clf()
    scatter3d(Y[:, 0], Y[:, 1], Y[:, 2], c=colors)
    save_many('exp_comb/mvu')

    #
    # Isomap
    #

    X = expand(N)
    Y, s = mds(X, 3)

    pylab.clf()
    pylab.plot(s[:10])
    save_many('exp_comb/isomab_scree')

    pylab.clf()
    scatter3d(Y[:, 0], Y[:, 1], Y[:, 2], c=colors)
    save_many('exp_comb/isomap')

