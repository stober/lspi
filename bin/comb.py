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
from utils import scatter3d
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
from isomap import shortest_paths


def expand(N):
    """ Compute the distances to 2 degree neighbors. """
    N = N + np.eye(49)
    N = np.dot(N, N)

    N = np.dot(N + np.eye(49), N + np.eye(49)) - 1.5 * N
    for i in range(N.shape[0]):
        N[i, i] = 0.0
    return N

if __name__ == '__main__':

    if True:
        walls = wall_pattern(9, 9)
        gw = Gridworld(nrows=9, ncols=9, walls=walls, endstates=[8])
        N = gw.neighbors([0, 2, 4, 6])
        pdb.set_trace()
        print shortest_paths(N)
        #N = expand(N)
        print N
        # gw.mainloop()
        filename = 'test.sdp'
        x, y = np.nonzero(N)
        indx = zip(x, y)
        size = N.shape[0]
        m = len(indx) + 1
        nblocks = 1
        c = [0.0]
        for i, j in indx:
            c.append(1.0)

        write_spda_file(filename, m, nblocks, size, c, indx)
        os.system("/usr/local/bin/csdp test.sdp test.sol")
        y, Z, X = read_sol_file('test.sol', size=49)
        u, s, v = la.svd(X)
        Y = u * np.sqrt(s)
        # pylab.plot(s[:10])
        scatter3d(Y[:, 0], Y[:, 1], Y[:, 2], show=True)
        pylab.show()

    if False:
        workspace = "/Users/stober/wrk/lspi/bin"

        walls = wall_pattern(9, 9)
        gw = GridworldGui(nrows=9, ncols=9, walls=walls, endstates=[0])
        # gw = Gridworld2(nrows=9, ncols=9, endstates=[0], walls=[], nrbf=15)
        #t = gw.trace(1000)
        t = gw.complete_trace()
        policy0 = np.zeros(gw.nfeatures())
        # w0, weights0 = LSPIRmax(t, 0.005, gw, policy0, method="dense", maxiter=1000, show=True, resample_epsilon=0.1, rmax=1000)
        w0, weights0 = LSPI(t, 0.005, gw, policy0, maxiter=100, method="sparse", show=True)

        # w0 = pickle.load(open("weights.pck"))
        pi = [gw.linear_policy(w0, s) for s in range(gw.nstates)]
        gw.set_arrows(pi)
        gw.background()
        pickle.dump(w0, open("weights.pck", "w"), pickle.HIGHEST_PROTOCOL)
        # gw.mainloop()
        # import pdb
        # pdb.set_trace()
        traces = gw.evaluate_policy(w0)
        pickle.dump(traces, open("comb_traces.pck", "w"), pickle.HIGHEST_PROTOCOL)
        # traces = pickle.load(open("traces.pck"))
        from dtw import edit_distance

        ematrix = np.zeros((49, 49))
        for (i, t) in enumerate(traces):
            for (j, s) in enumerate(traces):
                    ematrix[i, j] = edit_distance([e[1] for e in t], [l[1] for l in s])

        print ematrix
        from mds import mds
        y, s = mds(ematrix)
        from utils import scatter
        scatter(y[:, 0], y[:, 1])
        import pylab
        pylab.show()  # can't show if using pygame on Mac OSX
