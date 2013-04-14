#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TRACE_ANALYSIS.PY
Date: Tuesday, April  2 2013
Description: Plot trace data.
"""
import cPickle as pickle
import numpy as np
import pylab
from gridworld.gridworld8 import RBFObserverGridworld

def state_visitation_histogram(traces):
    hist = {}
    for t in traces:
        for s,a,r,ns,na in t:
            if s in hist:
                hist[s] += 1
            else:
                hist[s] = 1
        if ns in hist:
            hist[ns] += 1
        else:
            hist[ns] = 1
    return hist

def translate_states(hist):
    ogw = RBFObserverGridworld("/Users/stober/wrk/lspi/bin/16/20comp.npy", "/Users/stober/wrk/lspi/bin/16/states.npy", endstates = [0], walls=None, nrbf=80)
    result = {}

    for k,v in hist.items():
        coords = ogw.states[k]
        result[coords] = v

    return result

def plot_2d_histogram(hist,i):
    values = np.zeros((16,32))
    for k,v in hist.items():
        values[k[0],k[1]] = v
    #pylab.clf()
    pylab.imshow(values)
    pylab.gca().get_xaxis().set_ticks([])
    pylab.gca().get_yaxis().set_ticks([])
    pylab.xlabel("yaw")
    pylab.ylabel("pitch")
    pylab.savefig('hist{0}.png'.format(i))

for i in range(14):
    traces = pickle.load(open('traces{0}.pck'.format(i)))
    hist = state_visitation_histogram(traces)
    plot_2d_histogram(translate_states(hist),i)
    #print translate_states(hist)[(8,16)]



