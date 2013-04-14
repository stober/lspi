#!/usr/bin/env python
'''
@author jstober
'''

from gridworld.vergence import Vergence
from lspi import LSPI
import numpy as np
import pdb
from functools import partial
from dtw import non_dtw_distance
from mds import mds
import pylab

v = Vergence()
t = v.complete_trace()
policy0 = np.zeros(v.nfeatures())


w0, weights0 = LSPI(t, 0.00001, v, policy0, maxiter=100, method="sparse", show=True)
policy = partial(v.linear_policy, w0)

def cnt_rewards(t):
    rsum = 0
    for i in t:
        if len(i) == 0:
            rsum += 1  # bug fix for cases where state is endstate
        for j in i:
            rsum += j[2]
    return rsum

# for w in weights0:
for w in weights0:
    traces = v.evaluate_policy(w)
    print cnt_rewards(traces)
perfect = v.evaluate_func_policy(v.perfect_policy)
random = v.evaluate_func_policy(v.random_policy)
# print cnt_rewards(traces)
print cnt_rewards(perfect)
print cnt_rewards(random)

# test points

action_traces = []
for i in [91,93,95,97]:
    action_traces.append([j[1] for j in v.single_episode(policy, start=i)])

print action_traces

def costf(a,b):
    return np.abs(a-b)

n = len(action_traces)
ematrix = np.zeros((n, n))
for (i, t) in enumerate(action_traces):
    for (j, s) in enumerate(action_traces):
        ematrix[i, j] = non_dtw_distance(t,s,default=0, costf=costf)
    
y,s = mds(ematrix)
pylab.clf()
pylab.title("Sensorimotor Distances")
colors = ['red','orange','green','blue']
pts = []
for i in range(4):
    lbl = ""
    if i == 0:
        lbl = "Farthest Object"
    elif i == 3:
        lbl = "Closest Object"
    else:
        lbl = "None"
    print lbl
    pts.append(pylab.scatter([y[i,0]],[y[i,1]],c=colors[i],label=lbl))
# pylab.gca().set_xticklabels([])
pylab.gca().set_yticklabels([])
pylab.legend((pts[3],pts[0]), ("Closest Object", "Farthest Object"), scatterpoints=1)
#pylab.show()
pylab.savefig("vg_dist.pdf")



# print len(traces)
# print cnt_rewards(traces)
# for t in traces:
#     rsum = 0
#     for i in t:
#         rsum += i[2]
#     if rsum == 0:
#         print t


# for i in range(10):
#     t = v.trace(1000)
#     print cnt_rewards(t)

# for i in range(10):
#     t = v.trace(1000, policy=v.perfect_policy)
#     print cnt_rewards(t)

# test random policy for 1000 steps

# 146.0
# 115.0
# 104.0
# 82.0
# 64.0
# 98.0
# 135.0
# 91.0
# 85.0
# 116.0

# test lspi policy for 1000 steps (each policy iteration)

# 1.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 497.0
# 500.0

# 105 498
# 1.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 0.0
# 498.0
# 500.0
# 96.0
# 108.0
# 95.0
# 63.0
# 109.0
# 119.0
# 102.0
# 102.0
# 113.0
# 87.0

# overlay lspi over random
# compare random versus learned policy distance estimates

#(0,x) -> [1x] or [0,x]

# embed these traces


# for w in weights0:
#     print w
