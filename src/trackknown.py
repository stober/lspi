#!/usr/bin/env python 
'''
@author jstober

Simple class to track knowledge of states and actions. Based on 

L. Li, M. L. Littman, and C. R. Mansley, "Online exploration in least-squares policy iteration" AAMAS, 2009.
'''
import numpy as np
import pdb

class TrackKnown:
    """
    Track knowledge of states and actions.

    TODO: Generalize by adding epsilon and kd tree or approximation methods.
    """
    def __init__(self, nstates, nactions, mcount):
        self.nstates = nstates
        self.nactions = nactions
        self.mcount = mcount
        self.counts = np.zeros((nstates, nactions))

    def init(self, samples):
        for (s,a,r,ns,na) in samples:
            self.counts[s,a] += 1

    def uniq(self, samples):
        # this is not necessarily correct
        return list(set(samples))

    def resample(self, samples, trace, take_all=False):
        if take_all:
            for (s,a,r,ns,na) in trace:
                self.counts[s,a] += 1
                samples.append((s,a,r,ns,na))
        else:
            for (s,a,r,ns,na) in trace:
                if self.counts[s,a] < self.mcount:
                    self.counts[s,a] += 1
                    samples.append((s,a,r,ns,na))

    def known_pair(self,s,a):
        if self.counts[s,a] > self.mcount:
            return True
        else:
            return False

    def known_state(self,s):
        if np.greater(self.counts[s,:],self.mcount).all():
            return True
        else:
            return False

    def unknown(self,s):
        # indices of actions with low counts.
        return np.where(self.counts[s,:] < self.mcount)[0]

    def diagnostics(self):
        pass