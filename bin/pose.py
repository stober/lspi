#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: POSE.PY
Date: Thursday, April  4 2013
Description: Matching pose experiment.
"""

from gridworld.gridworld8 import MultiTargetGridworld
from lspi import LSPI
import numpy as np
from functools import partial
import pdb
import cPickle as pickle

def callback(iters, current, env):
    print 'Callback: ', iters

    # get all traces
    traces = env.evaluate_policy(current)
    perfect = env.evaluate_func_policy(env.perfect_policy)

    pickle.dump((traces,perfect), open('pose/pose_traces{0}.pck'.format(iters),'w'), pickle.HIGHEST_PROTOCOL)

    p_avg_reward = 0
    p_avg_length = 0
    for t in perfect:
        p_avg_reward += sum([i[2] for i in t])
        p_avg_length += len(t)
    p_avg_reward = p_avg_reward / float(len(perfect))
    p_avg_length = p_avg_length / float(len(perfect))
    print "Perfect: ", p_avg_reward
    print "Perfect: ", p_avg_length

    # measure task performance
    avg_reward = 0
    avg_length = 0
    for t in traces:
        avg_reward += sum([i[2] for i in t])
        avg_length += len(t)
    avg_reward = avg_reward / float(len(traces))
    avg_length = avg_length / float(len(traces))
    print 'Avg reward: ', avg_reward
    print "Avg length: ", avg_length
    pickle.dump((avg_reward,avg_length,p_avg_reward,p_avg_length), open('pose/pose_reward{0}.pck'.format(iters),'w'), pickle.HIGHEST_PROTOCOL)


gw = MultiTargetGridworld(nrows = 10, ncols = 10)
t = gw.complete_trace()

policy0 = np.zeros(gw.nfeatures())
w0, weights0 = LSPI(t, 0.0001, gw, policy0, maxiter=100, method="sparse", show=True, callback=callback)

policy = partial(gw.linear_policy, w0)
#pdb.set_trace()
t2 = gw.trace(1000, policy=policy, reset_on_endstate=True)
print t2
