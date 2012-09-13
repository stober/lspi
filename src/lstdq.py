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
from multiprocessing import Pool, Queue, Process, log_to_stderr, SUBDEBUG, cpu_count
from utils import sp_create,chunk

#logger = log_to_stderr()
#logger.setLevel(SUBDEBUG)

# lsqr return values
# x : ndarray of float
# istop : int
# itn : int
# r1norm : float
# r2norm : float
# anorm : float
# acond : float
# arnorm : float
# xnorm : float
# var : ndarray of float

def solve(A, b, method="pinv"):
    info = {}
    w = None

    if method == "pinv":
        info['acond'] = la.cond(A)
        w = np.dot(la.pinv(A),b)

    elif method == "dot":
        info['acond'] = la.cond(A)
        w = A.dot(b).toarray()[:,0]

    elif method == "lsqr":
        squeeze_b = np.array(b.todense()).squeeze()
        qr_result = spla.lsqr(A,squeeze_b.T, atol=1e-8, btol=1e-8, show=True)
    
        # diagnostics are already computed so we always populate info in this case    
        w = qr_result[0]
        info['istop'] = qr_result[1]
        info['itn'] = qr_result[2]
        info['r1norm'] = qr_result[3]
        info['r2norm'] = qr_result[4]
        info['anorm'] = qr_result[5]
        info['acond'] = qr_result[6]
        info['xnorm'] = qr_result[7]
        info['var'] = qr_result[8]
    
    else:
        raise ValueError, "Unknown solution method!"

    return w,info

def LSTDQ(D,env,w,damping=0.001,testing=False):
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

    w,info = solve(A,b,method="pinv")
    return A,b,w,info

import scipy.sparse as sp
import scipy.sparse.linalg as spla

def FastLSTDQ(D,env,w,damping=0.001,testing=False,format="csr",child=False):
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

    w, info = solve(A,b,method="lsqr")
    
    if testing == True: 
        print "Testing against dense version."
        dA,db,dw = LSTDQ(D,env,w,damping=damping)        

        if not np.allclose(dA,A.todense()):
            print "***** (dA,A) are not CLOSE! *****"
        else:
            print "(dA,A) are close!"

        if not np.allclose(b.T.todense(),db):
            print "****** (db,b) are not CLOSE! *****"
        else:
            print "(db,b) are close!"

        if not np.allclose(w, dw):
            print "****** (dw,w) are not CLOSE! *****"
        else:
            print "(dw,w) are close!"

    return A,b,w,info


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

def AltLSTDQ(D,env,w,damping=0.001,testing=False,format="csr",ncpus=None):
    """Alternative parallel implementation. Based on some intial tests the other version is faster. """

    if ncpus:
        nprocess = ncpus
    else:
        nprocess = cpu_count()
    pool = Pool(nprocess,initializer=child_initialize,initargs=(env,w,format))

    k = -1
    k = len(w)

    A = sp.identity(k,format=format) * damping
    b = sp_create(k,1,format)

    it = pool.imap_unordered(child_compute,D,100)
    for T,t in it:
        A = A + T
        b = b + t

    w,info = solve(A,b,method="lsqr")
    return A,b,w,info


def ParallelLSTDQ(D,env,w,damping=0.001,testing=False,format="csr",ncpus=None):
    """
    D : source of samples (s,a,r,s',a')
    env: environment contianing k,phi,gamma
    w : weights for the linear policy evaluation

    Note that "csr" format seems to work best. Should convert to csr for arithmatic operations automatically.
    """

    if ncpus:
        nprocess = ncpus
    else:
        nprocess = cpu_count()
    pool = Pool(nprocess)
    indx = chunk(len(D),nprocess)
    results = []
    for (i,j) in indx:
        r = pool.apply_async(FastLSTDQ,(D[i:j],env,w,damping,testing,format,True))
        results.append(r)
        
    k = len(w)
    A = sp.identity(k,format=format)
    b = sp_create(k,1,format)
    for r in results:
        T,t = r.get()
        A = A + T
        b = b + t

    w,info = solve(A,b,method="lsqr")
    return A,b,w,info

def OptLSTDQ(D,env,w,damping=0.001,testing=True,format="csr"):
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
        dA,db,dw = LSTDQ(D,env,w,damping=damping) 

        dB = la.pinv(dA)
        if not np.allclose(dB,B.todense(),atol=1e-6,rtol=0.0):
            print "***** (dB,B) are not CLOSE! *****"
        else:
            print "(dB,B) are close!"

        if not np.allclose(b.T.todense(),db):
            print "****** (db,b) are not CLOSE! *****"
        else:
            print "(db,b) are close!"

    w,info = solve(B,b,method="dot")
    return B,b,w,info

if __name__ == '__main__':

    pass
