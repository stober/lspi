#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: PLOT.PY
Date: Saturday, March 23 2013
Description: plot stuff
"""

import cPickle as pickle
import bz2
import pylab
import pylab
from isomap import isomap
from mds import mds
import numpy as np
from dtw import edit_distance_vc, non_dtw_distance
from gridworld.gridworld8 import RBFObserverGridworld
from utils import create_norm_colors
from procrustes import procrustes

def adist(a1, a2):
    d1 = np.zeros((8,8))
    d2 = np.zeros((9,9))

    for i in range(8):
        d1[i,(i + 1) % 8] = 0.5
        d1[i,(i + 2) % 8] = 1.0
        d1[i,(i + 3) % 8] = 1.5
        d1[i,(i + 4) % 8] = 2.0
        d1[i,(i + 5) % 8] = 1.5
        d1[i,(i + 6) % 8] = 1.0
        d1[i,(i + 7) % 8] = 0.5

    d2[0:8,0:8] = d1
    for i in range(8):
        d2[i,8] = 1.0

    d2[8,1] = 1.0
    d2[8,2] = 1.0
    d2[8,3] = 1.0
    d2[8,4] = 1.0
    d2[8,5] = 1.0
    d2[8,6] = 1.0
    d2[8,7] = 1.0
    d2[8,8] = 0.0 # should not happen

    return d2[a1,a2]



nearby_states = []
errs = []
areward = []
projections = []
for i in range(13):
    print i
    fp = open('misc{0}.pck'.format(i))
    stuff = pickle.load(fp)
    (iters, err, gt, avg_reward, current, y, s, adj) = stuff 
    errs.append(err)
    areward.append(avg_reward)
    projections.append(y)

if False: 

    ogw = RBFObserverGridworld("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = [272], walls=None, nrbf=80)
    pts = np.array(ogw.states.values())
    colors = create_norm_colors(pts)

    pylab.scatter(pts[:, 0], pts[:, 1], c=colors)
    pylab.title("Ground Truth")
    pylab.savefig('gt.pdf')

    for k in range(13):

        # traces = pickle.load(open('traces{0}.pck'.format(k)))
        
        # # find current embedding
        # ematrix = np.zeros((512,512))        
        # for (i,t) in enumerate(traces):
        #     for (j,s) in enumerate(traces):
        #         #ematrix[i,j] = edit_distance_vc([e[1] for e in t], [l[1] for l in s], (1.0, 1.0, 1.5))
        #         ematrix[i,j] = non_dtw_distance([e[1] for e in t], [l[1] for l in s], default = 8, costf=adist)

        # pickle.dump(ematrix,open('ematrix_revised{0}'.format(k),'w'),pickle.HIGHEST_PROTOCOL)
        ematrix = pickle.load(open('ematrix_revised{0}'.format(k)))
        
        y,s = mds(ematrix)
        pylab.clf()
        pylab.title("Iteration {0}".format(k))
        pylab.scatter(y[:,0],y[:,1],c=colors)
        pylab.savefig("embed_{0}".format(k))

if False:
    traces = pickle.load(open('traces13.pck'))
    # find current embedding
    ematrix = np.zeros((512,512))        
    for (i,t) in enumerate(traces):
        for (j,s) in enumerate(traces):
            #ematrix[i,j] = edit_distance_vc([e[1] for e in t], [l[1] for l in s], (1.0, 1.0, 1.5))
            ematrix[i,j] = non_dtw_distance([e[1] for e in t], [l[1] for l in s], default = 8, costf=adist)
    y,s = mds(ematrix)#isomap(ematrix)
    pylab.scatter(y[:,0],y[:,1])
    pylab.show()


if False:
    for (i, y) in enumerate(projections):
        pylab.clf()
        pylab.title("Iteration {0}".format(i))
        pylab.scatter(y[:,0],y[:,1])
        pylab.savefig("embed_{0}".format(i))

if False:
    pylab.clf()
    pylab.title("Embedding Quality")
    pylab.ylabel("Procrustes Error")
    pylab.xlabel("LTDQ Iterations")
    pylab.plot(errs, label="SE Error")
    #pylab.show()

    pca = [0.87542833428] * len(errs)
    pylab.plot(pca, c='red',label="PCA Error")
    pylab.legend(loc=3)
    pylab.savefig("pe.pdf")


if False:
    pylab.title("Policy Improvement")
    pylab.ylabel("Average Reward")
    pylab.xlabel("LTDQ Iterations")
    pylab.plot(areward)
    pylab.ylim([0,1.1])
    pylab.savefig("re.pdf")

if True:
    ogw = RBFObserverGridworld('/Users/stober/wrk/lspi/bin/16/20comp.npy', '/Users/stober/wrk/lspi/bin/16/states.npy', endstates = [272], walls=None, nrbf=80)
    pts = np.array(ogw.states.values())
    colors = create_norm_colors(pts)

    comps = np.load('/Users/stober/wrk/lspi/bin/16/5comp.npy')
    
    print procrustes(pts, comps[:,:2])

    pylab.clf()
    pylab.title('PCA Embedding')
    pylab.scatter(comps[:,0], comps[:,1],c=colors)
    pylab.xlabel('First Component')
    pylab.ylabel('Second Component')
    pylab.savefig('pca_embedding.pdf')


    pylab.clf()
    pylab.title('Ground Truth')
    pylab.scatter(pts[:,0], pts[:,1], c=colors)
    pylab.xlabel('Yaw')
    pylab.ylabel('Roll')
    pylab.savefig('gt_embedding.pdf')


    #pcs, m, s, T, u = pickle.load(bz2.BZ2File("/Users/stober/wrk/ros/gazebo_objects/bin/16/pcs.pck"))
    s = np.load('s.npy')
    s = s / np.sum(s)

    pylab.clf()
    pylab.plot(s)
    pylab.title('Scree Plot')
    pylab.xlabel('Component #')
    pylab.ylabel('% Weight')
    pylab.xlim([0,100])
    pylab.savefig('pca_scree.pdf')



    pylab.clf()
    pylab.plot(s,c='red',label='PCA Weights')


    ematrix = pickle.load(open('ematrix_revised12'))
    y,s = mds(ematrix)
    s = s / np.sum(s)
    pylab.plot(s, c='blue',label='SE Weights')
    pylab.title('Scree Plot')
    pylab.xlabel('Component #')
    pylab.ylabel('% Weight')
    pylab.xlim([0,20])
    pylab.legend()
    pylab.savefig('combined_scree.pdf')


    
