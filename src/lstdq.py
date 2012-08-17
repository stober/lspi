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
def LSTDQ(D,env,w,damping=0.001):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    """

    k = -1
    k = len(w)

    #A = np.eye(k) * 0.001
    A = np.zeros((k,k)) + np.eye(k) * damping
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

    print "DET: ", la.det(A)
    if la.det(A) == 0.0:
        print "WARNING: A is singular!"
    return A,b,np.dot(la.pinv(A), b)

import scipy.sparse as sp
import scipy.sparse.linalg as spla

@timerflag
@debugflag
def QR_LSTDQ(D,env,w):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """

    testing=True

    k = -1
    k = len(w)

    #A = np.eye(k) * 0.001
    #A = np.zeros((k,k))
    A = sp.eye(k,k) * 0.001 #.dok_matrix((k,k))
    b = sp.dok_matrix((k,1))

    dA = None
    db = None
    if testing == True:
        dA = np.zeros((k,k))
        db = np.zeros(k)

    i = 0
    for (s,a,r,ns,na) in D:

        #print i
        i += 1

        features = env.phi(s,a,sparse=True)

        # we may want to evaluate policies whose features are
        # different from ones that can express the true value
        # function, e.g. tabular

        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next,sparse=True)

        nf = features - env.gamma * newfeatures
        T = sp.kron(features, nf.T)
        A = A + T
        b = b + features * r 

        if testing == True:
            """
            Do expensive dense computations and compare against the sparse computations for testing.
            """

            d_features = env.phi(s,a)
            d_features_new = env.phi(ns,next)
            dT = np.outer(d_features, d_features - env.gamma * d_features_new)

            if not np.allclose(dT, T.todense()):
                print "****** (dT,T) are not CLOSE! ******"

            dA = dA + dT
            db = db + d_features * r

            #print "DET: ", la.det(dA)
            #if la.det(dA) == 0.0:
                #print "WARNING: A is singular!"

    squeeze_b = np.array(b.todense()).squeeze()
    stuff = spla.lsqr(A,squeeze_b.T,atol=1e-8,btol=1e-8,show=True)
    

    if testing == True: 

        if not np.allclose(dA,A.todense()):
            print "***** (dA,A) are not CLOSE! *****"


        if not np.allclose(b.T.todense(),db):
            print "****** (db,b) are not CLOSE! *****"

        dw = np.dot(la.pinv(dA),db)

        print "***** Weight Diff *****"
        print la.norm(dw - stuff[0])

    return A,b,stuff[0]



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
