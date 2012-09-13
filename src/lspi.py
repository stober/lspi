#!/usr/bin/python
"""
Author: Jeremy M. Stober
Program: LSPI.PY
Date: Tuesday, January 12 2010
Description: Least Squares Policy Iteration from Lagoudakis and Parr. 2003. JMLR.
"""

import pdb
import numpy as np
import numpy.random as npr
import random as pr
import numpy.linalg as la
import sys
import pickle
from lstdq import *
from utils import debugflag, timerflag, consumer
import scipy.sparse as sp

class Diagnostics:
    """
    A class to track the algorithm performance.
    """

    def __init__(self,env):
        self.env = env
        self.Q = np.zeros((env.nstates,env.nactions))
        self.Qp = np.zeros((env.nstates,env.nactions))
        self.previous = np.zeros(env.nfeatures())

    def __call__(self,iters, policy, A):
            for i in range(self.env.nstates):
                for j in range(self.env.nactions):
                    self.Q[i,j] = np.dot(self.env.phi(i,j), policy)
        
            if sp.issparse(A):
                [u,s,v] = sp.linalg.svds(A)
            else:
                [u,s,v] = la.svd(A)
            result = """
            Iteration {0}
            Weight Sum {1}
            Weight Diff {2}
            Value Diff {3}
            Max Singular Value {4}
            Min Singular Value {5}
            """.format(iters,np.sum(policy),la.norm(policy-self.previous),la.norm(self.Q - self.Qp),np.max(s),np.min(s))
        
            self.previous[:] = policy[:]
            self.Qp[:] = self.Q[:]

            # Callback into the environment class for gui stuff.
            if hasattr(self.env,'callback'):
                self.env.callback(iters, policy)
            return result 
    

@timerflag
@debugflag
def LSPI(D, epsilon, env, policy0, method="dense", save=False, maxiter=10, show=False, format = "csr", ncpus=None):

    current = policy0
    all_policies = [current]

    if show:
        diagnostics = Diagnostics(env)

    if save:
        fp = open('weights.pck','w')

    iters = 0
    finished = False
    while iters < maxiter and not finished:

        all_policies.append(current)

        if method is "dense":
            A,b,current,info = LSTDQ(D, env, current)
        elif method is "sparse":
            A,b,current,info = FastLSTDQ(D, env, current, format=format)
        elif method is "opt":
            A,b,current,info = OptLSTDQ(D, env, current, format=format)
        elif method is "parallel":
            A,b,current,info = ParallelLSTDQ(D, env, current, format=format, ncpus=ncpus)
        elif method is "alt":
            A,b,current,info = AltLSTDQ(D, env, current, format=format, ncpus=ncpus)
        else:
            raise ValueError, "Unknown method!"

        if save:
            pickle.dump(current,fp,pickle.HIGHEST_PROTOCOL)

        if show:
            print diagnostics(iters,current,A)
            for (i,p) in enumerate(all_policies):
                print "policy: ", i, la.norm(p - current)

        for p in all_policies:
            if la.norm(p - current) < epsilon:  
                finished = True
        
        iters += 1

    return current, all_policies

def test():
    try:
        from gridworld.chainwalk import Chainwalk
    except:
        print "Unable to import Chainwalk for test!"
        return

    cw = Chainwalk()
    trace = cw.trace()
    zeros = np.zeros(cw.nfeatures())
    w = LSPI(trace,0.0001,cw,zeros,show=True)
    print w

if __name__ == '__main__':

    test()
