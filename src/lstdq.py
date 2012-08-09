#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: LSTDQ.PY
Date: Wednesday, December 16 2009
Description: LSTDQ implementation from Lagoudakis and Parr. 2003. Least-Squares Policy Iteration. Journal of Machine Learning Research.
"""

import os, sys, getopt, pdb, string

import numpy as np
import numpy.random as npr
import random as pr
import numpy.linalg as la
from utils import debugflag, timerflag


@timerflag
@debugflag
def LSTDQ(D,env,w):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """

    k = -1
    k = len(w)

    A = np.zeros((k,k))
    b = np.zeros(k)

    i = 0
    for (s,a,r,ns,na) in D:

        #print i
        i += 1

        features = env.phi(s,a)

        # we may want to evaluate policies whose features are
        # different from ones that can express the true value
        # function, e.g. tabular

        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next)

        A = A + np.outer(features, features - env.gamma * newfeatures)
        b = b + features * r

    return A,b,np.dot(la.pinv(A), b)

import scipy.sparse as sp
import scipy.sparse.linalg as spla

@timerflag
@debugflag
def OptLSTDQ(D,env,w):
    """
    Use paper's suggested optimization method.

    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """

    k = -1
    k = len(w)

    B = sp.identity(k)
    b = sp.dok_matrix((k,1))

    i = 0
    for (s,a,r,ns,na) in D:

        i += 1

        features = env.phi(s,a,sparse = True)

        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse = True)

        uv = np.dot(features,newfeatures.T)
        N = np.dot(np.dot(B,uv),B)
        d = 1 + np.dot(np.dot(newfeatures.T,B),features)[0,0]

        B = B - N / d
        b = b + features * r

    return B,b,np.dot(B,b)

@timerflag
@debugflag
def FastLSTDQ(D,env,w):
    """
    Employ as many tricky speedups as I can for large (sparse) phi.

    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """

    k = -1
    k = len(w)

    A = sp.dok_matrix((k,k))
    b = sp.dok_matrix((k,1))

    i = 0
    for (s,a,r,ns,na) in D:

        i += 1

        features = env.phi(s,a,sparse=True)

        # we may want to evaluate policies whose features are
        # different from ones that can express the true value
        # function, e.g. tabular

        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse = True)

        for (i,j) in features.iterkeys():
            for (s,t) in newfeatures.iterkeys():
                A[i,s] += features[i,j] * (features[s,t] - env.gamma * newfeatures[s,t])

            b[i,j] += features[i,j] * r

        # A = A + np.dot(features,(features - env.gamma * newfeatures).T)
        # b = b + features * r

    # TODO : Not sure what solver method to use here.
    # return spla.spsolve(A,b)

	A = A.tocsr()
    b = np.array(b.todense()).squeeze() # matrix squeeze seems to be broken

    # Note: If the damping parameter is too large that part of the
    # optimization problem will dominate the solution resulting in a
    # poor weight estimate.

    #stuff = spla.lsmr(A,b.T,atol=1e-8,btol=1e-8,show=True)
    stuff = spla.lsqr(A,b.T,atol=1e-8,btol=1e-8,damp=1e-6,show=True)
    #stuff = spla.lsqr(A,b.T,atol=1e-8,btol=1e-8,show=True)

    return A,b,stuff[0]


if __name__ == '__main__':

    pass
