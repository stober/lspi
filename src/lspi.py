#!/usr/bin/python
"""
Author: Jeremy M. Stober
Program: LSPI.PY
Date: Tuesday, January 12 2010
Description: Least Squares Policy Iteration from Lagoudakis and Parr. 2003. JMLR.
"""

import numpy as np
import numpy.random as npr
import random as pr
import numpy.linalg as la
import sys
import pickle
from lstdq import *

def LSPI(D, epsilon, env, policy0, save=False):

    current = policy0
    all_policies = [current]
    if save:
        fp = open('weights.pck','w')

    iter = 0
    while True:

        print "Iteration %d" % iter, " Weight Sum: %f" % np.sum(current)

        previous = current
        current = LSTDQ(D, env, previous)
        all_policies.append(current)

        if save:
            pickle.dump(current,fp,pickle.HIGHEST_PROTOCOL)

        print la.norm(previous - current)
        if la.norm(previous - current) < epsilon: break

        iter += 1

        # callback into the environment class for gui stuff?
        env.callback(iter, current)

    return current, all_policies

def test():
    try:
        from gridworld.chainwalk import Chainwalk
    except:
        print "Unable to import Chainwalk for test!"
        return

    cw = Chainwalk()
    trace = cw.trace()
    zeros = np.zeros(cw.nfeatures)
    w = LSPI(trace,0.0001,cw,zeros)
    print w

if __name__ == '__main__':

    test()