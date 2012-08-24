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
from utils import debugflag, timerflag, sp_create

@timerflag
@debugflag
def LSTDQ(D,env,w,damping=0.001,show=False,testing=False):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    """

    k = -1
    k = len(w)

    #A = np.eye(k) * 0.001
    A = np.eye(k) * damping
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
def FastLSTDQ(D,env,w,damping=0.001,show=False,testing=False,format="dok"):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation

    Note that "dok" format seems to work best. Should convert to csr for arithmatic operations automatically.
    """

    k = -1
    k = len(w)

    #A = np.eye(k) * 0.001
    #A = np.zeros((k,k))
    A = sp.identity(k,format=format) * damping
    b = sp_create(k,1,format)

    i = 0
    for (s,a,r,ns,na) in D:

        i += 1

        features = env.phi(s, a, sparse=True, format=format)

        # we may want to evaluate policies whose features are
        # different from ones that can express the true value
        # function, e.g. tabular

        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse=True, format=format)

        nf = features - env.gamma * newfeatures
        T = sp.kron(features, nf.T)
        A = A + T
        b = b + features * r 

    squeeze_b = np.array(b.todense()).squeeze()
    stuff = spla.lsqr(A,squeeze_b.T,atol=1e-8,btol=1e-8,show=show)
    
    if testing == True: 
        print "Testing against dense version."
        dA,db,dw = LSTDQ(D,env,w,damping=damping,show=show)        

        if not np.allclose(dA,A.todense()):
            print "***** (dA,A) are not CLOSE! *****"
        else:
            print "(dA,A) are close!"

        if not np.allclose(b.T.todense(),db):
            print "****** (db,b) are not CLOSE! *****"
        else:
            print "(db,b) are close!"

        if not np.allclose(stuff[0], dw):
            print "****** (dw,w) are not CLOSE! *****"
        else:
            print "(dw,w) are close!"

    return A,b,stuff[0]



@timerflag
@debugflag
def OptLSTDQ(D,env,w,damping=0.001,show=False,testing=True,format="csr"):
    """
    Use paper's suggested optimization method.

    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    """

    k = -1
    k = len(w)


    B = sp.identity(k,format=format) * 1.0/damping
    b = sp_create(k,1,format)

    i = 0
    for (s,a,r,ns,na) in D:

        i += 1

        features = env.phi(s,a,sparse = True, format=format)

        next = env.linear_policy(w, ns)
        newfeatures = env.phi(ns, next, sparse = True, format=format)

        nf = features - env.gamma * newfeatures
        uv = sp.kron(features,nf.T)
        N = B.dot(uv).dot(B)
        d = 1 + nf.T.dot(B).dot(features)[0,0]

        B = B - N / d
        b = b + features * r

    if testing:
        print "Testing against dense version."
        dA,db,dw = LSTDQ(D,env,w,damping=damping,show=show) 

        dB = la.pinv(dA)
        if not np.allclose(dB,B.todense(),atol=1e-6,rtol=0.0):
            print "***** (dB,B) are not CLOSE! *****"
        else:
            print "(dB,B) are close!"

        if not np.allclose(b.T.todense(),db):
            print "****** (db,b) are not CLOSE! *****"
        else:
            print "(db,b) are close!"


    return B,b,B.dot(b).toarray()[:,0]


if __name__ == '__main__':

    pass
