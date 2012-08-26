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
from multiprocessing import Pool, Queue, Process, log_to_stderr, SUBDEBUG
from utils import sp_create

#logger = log_to_stderr()
#logger.setLevel(SUBDEBUG)

def LSTDQ(D,env,w,damping=0.001,show=False,testing=False):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation
    damping : keeps the result relatively stable (solves some difficulties with oscillation if A is singular)
    """

    k = -1
    k = len(w)

    A = np.eye(k) * damping
    b = np.zeros(k)

    for (s,a,r,ns,na) in D:

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

def FastLSTDQ(D,env,w,damping=0.001,show=False,testing=False,format="csr",child=False):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation

    Note that "csr" format seems to work best. Should convert to csr for arithmatic operations automatically.
    """

    k = -1
    k = len(w)

    A = sp.identity(k,format=format) * damping
    b = sp_create(k,1,format)

    for (s,a,r,ns,na) in D:

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

    if child: # used in parallel implementation!
        return A,b

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


class ChildData:

    def __init__(self,env,w,format):
        self.env = env
        self.w = w
        self.format = format

def child_initialize(env,w,format):
    # initialize some global data for child process
    global child
    child = ChildData(env,w,format)

def child_compute(args):
    s,a,r,ns,na = args
    
    # get initialized global data
    global child
    env = child.env
    w = child.w
    format = child.format
    
    features = env.phi(s,a,sparse=True,format=format)
    next = env.linear_policy(w,ns)
    newfeatures = env.phi(ns, next, sparse=True, format=format)
    nf = features - env.gamma * newfeatures
    T = sp.kron(features,nf.T)
    t = features * r
    return T,t 

def AltLSTDQ(D,env,w,damping=0.001,show=False,testing=False,format="csr"):
    """Alternative parallel implementation. Based on some intial tests the other version is faster. """

    nprocess = 4
    pool = Pool(nprocess,initializer=child_initialize,initargs=(env,w,format))

    k = -1
    k = len(w)

    A = sp.identity(k,format=format) * damping
    b = sp_create(k,1,format)

    it = pool.imap_unordered(child_compute,D,100)
    for T,t in it:
        A = A + T
        b = b + t

    squeeze_b = np.array(b.todense()).squeeze()
    stuff = spla.lsqr(A,squeeze_b.T,atol=1e-8,btol=1e-8,show=show)
    
    return A,b,stuff[0]


def ParallelLSTDQ(D,env,w,damping=0.001,show=False,testing=False,format="csr"):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation

    Note that "csr" format seems to work best. Should convert to csr for arithmatic operations automatically.
    """

    nprocess = 4 # TODO: make this the cpu count
    pool = Pool(nprocess)

    chunk = len(D) / nprocess
    indx = [[i * chunk, ((i+1) * chunk)] for i in range(nprocess)]
    indx[-1][1] = len(D)
    results = []
    for (i,j) in indx:
        r = pool.apply_async(FastLSTDQ,(D[i:j],env,w,damping,show,testing,format,True))
        results.append(r)
        
    k = len(w)
    A = sp.identity(k,format=format)
    b = sp_create(k,1,format)
    for r in results:
        T,t = r.get()
        A = A + T
        b = b + t

    squeeze_b = np.array(b.todense()).squeeze()
    stuff = spla.lsqr(A,squeeze_b.T,atol=1e-8,btol=1e-8,show=show)
    return A,b,stuff[0]

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

    for (s,a,r,ns,na) in D:

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
