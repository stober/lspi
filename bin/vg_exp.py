#!/usr/bin/env python
'''
@author jstober
'''

from gridworld.vergence import Vergence
from lspi import LSPI
import numpy as np
import pdb
from functools import partial

v = Vergence()
t = v.trace(1000)
policy0 = np.zeros(v.nfeatures())


w0, weights0 = LSPI(t, 0.00001, v, policy0, maxiter=100, method="sparse", show=True)
policy = partial(v.linear_policy, w0)
t2 = v.trace(1000,policy=policy)

def cnt_rewards(t):
    rsum = 0
    for i in t:
        rsum += i[2]
    return rsum

# show reward transitions in the trace
tcnt = 0
for i in t:
    if i[2] == 1.0:
        tcnt +=1 

t2cnt = 0
for i in t2:
    if i[2] == 1.0:
        t2cnt += 1

print tcnt, t2cnt # 109, 499
# print len(weights0)
# perf = []
# import pdb
# pdb.set_trace()
# perf.append(v.evaluate_policy(w0))
for w in weights0:
    policy = partial(v.linear_policy, w)
    t = v.trace(1000, policy=policy)
    print cnt_rewards(t)
    #perf.append(v.evaluate_policy(w))

for i in range(10):
    t = v.trace(1000)
    print cnt_rewards(t)

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
