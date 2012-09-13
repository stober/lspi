#!/usr/bin/env python
'''
@author jstober
'''

import scipy.sparse as sp
import numpy as np
import numpy.linalg as la
import cPickle as pickle
from gridworld.gridworldgui import GridworldGui
import scipy.linalg as sla
import scipy.sparse.linalg

dA = np.load('dA.npy')
sA = np.load('sA.npy')
db = np.load('db.npy')
sb = np.load('sb.npy')
dw = np.load('dw.npy')
sw = np.load('sw.npy')

D = pickle.load(open("lspi_trace.pck"))

def test(D):
	k = len(dw)
	w = np.zeros(k)
	gw = GridworldGui(nrows = 9, ncols = 9, endstates = [0], walls = [])
	A1 = np.eye(k) * 0.001
	b1 = np.zeros(k)
	for (s,a,r,ns,na) in D:
		
		features = gw.phi(s,a)
		next = gw.linear_policy(w,ns)
		newfeatures = gw.phi(ns,next)

		# dense computation
		A1 = A1 + np.outer(features, features - gw.gamma * newfeatures)
		b1 = b1 + features * r

	return A1,b1 

print np.allclose(dw,sw)
aindx = np.argmax(np.abs(dw - sw))
print aindx, dw[aindx], sw[aindx]
print la.norm(dw), la.norm(sw)

print la.norm(np.dot(dA,sw) - db)
print la.norm(np.dot(dA,dw) - db)

x = sla.solve(dA,db)
print la.norm(np.dot(dA,x) - db)
# y = scipy.sparse.linalg.spsolve(sp.dA,db)
# print la.norm(np.dot(dA,db) - db)

# print sb.shape, db.shape
# print np.allclose(sb.T,db)
# print np.max(db - sb.T)
# print np.max(dA - sA)

# print dA.shape, sA.shape

# maxindx =  np.argmax(dA - sA)
# indx = np.unravel_index(maxindx, dA.shape)
# print dA[indx], sA[indx]

# A1,b1 = test(D)
# print np.allclose(dA,A1)
# aindx = np.argmax(dA-A1)
# indx = np.unravel_index(aindx, dA.shape)
# print indx, dA[indx],A1[indx]
# print np.allclose(sA,A1)
# print np.allclose(db,b1)
