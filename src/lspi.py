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
from utils import debugflag, timerflag
import scipy.sparse as sp


@timerflag
@debugflag
def FastLSPI(D, epsilon, env, policy0, save = False, maxiter=10):

    current = policy0
    k = len(policy0)
    A = sp.dok_matrix((k,k))
    Q = np.zeros((env.nstates,env.nactions))
    all_policies = [current]
    if save:
        fp = open('weights.pck','w')

    iter = 0
    while iter < maxiter:

        print "Iteration %d" % iter, " Weight Sum: %f" % np.sum(current)

        previous = current.copy()
        Ap = A.copy()
        Qp = Q.copy()

        A,b,current = FastLSTDQ(D, env, previous)
        all_policies.append(current)

        if save:
            pickle.dump(current,fp,pickle.HIGHEST_PROTOCOL)

        #pdb.set_trace()

        print "WEIGHT DIFF:", la.norm(previous - current)
        print "MODEL DIFF:", la.norm((Ap - A).todense())

        # compute s,a matrix of values
        for i in range(env.nstates):
            for j in range(env.nactions):
                Q[i,j] = np.dot(env.phi(i,j),current)

        print "VALUE DIFF:", la.norm(Qp - Q)

        if la.norm(previous - current) < epsilon: break

        iter += 1

        # callback into the environment class for gui stuff?
        env.callback(iter, current)

    return current, all_policies


@timerflag
@debugflag
def QR_LSPI(D, epsilon, env, policy0, save=False, maxiter=10):

    current = policy0
    all_policies = [current]
    if save:
        fp = open('weights.pck','w')

    iter = 0
    while iter < maxiter:

        print "Iteration %d" % iter, " Weight Sum: %f" % np.sum(current)

        previous = current
        A,b,current = QR_LSTDQ(D, env, previous)
        all_policies.append(current)

        if save:
            pickle.dump(current,fp,pickle.HIGHEST_PROTOCOL)

        print "WEIGHT DIFF:", la.norm(previous - current)
        if la.norm(previous - current) < epsilon: break

        iter += 1

        # callback into the environment class for gui stuff?
        env.callback(iter, current)

    return current, all_policies



@timerflag
@debugflag
def LSPI(D, epsilon, env, policy0, save=False, maxiter=10):

    current = policy0
    all_policies = [current]
    Q = np.zeros((env.nstates,env.nactions))
    Qp = np.zeros((env.nstates,env.nactions))

    if save:
        fp = open('weights.pck','w')

    iter = 0
    while iter < maxiter:

        print "Iteration %d" % iter, " Weight Sum: %f" % np.sum(current)

        previous = current
        A,b,current = LSTDQ(D, env, previous)
        all_policies.append(current)

        if save:
            pickle.dump(current,fp,pickle.HIGHEST_PROTOCOL)

        print "WEIGHT DIFF:", la.norm(previous - current)
        if la.norm(previous - current) < epsilon: break

        # compute s,a matrix of values
        for i in range(env.nstates):
            for j in range(env.nactions):
                Q[i,j] = np.dot(env.phi(i,j),current)

        print "VALUE DIFF:", la.norm(Qp - Q)
        
        if iter > 10:
            pdb.set_trace()

        Qp[:] = Q[:]



        iter += 1

        # callback into the environment class for gui stuff?
        env.callback(iter, current)

    return current, all_policies

def test():
    try:
        from gridworld.chainwalk import Chainwalk
    except:
        print "Unable to import Chainwalk for test!"
        return

    cw = Chainwalk()
    trace = cw.trace()
    zeros = np.zeros(cw.nfeatures)
    w = LSPI(trace,0.0001,cw,zeros)
    print w

if __name__ == '__main__':

    test()
