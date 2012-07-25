#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: TEST_LSPI.PY
Date: Wednesday, July 25 2012
Description: Test code for LSPI.
"""


import numpy as np
from gridworld.chainwalk import Chainwalk
from lspi import LSTDQ

cw = Chainwalk()
t = cw.trace(1000)
policy0 = np.zeros(cw.nfeatures)
import pdb
pdb.set_trace()
print LSTDQ(t, cw, policy0)


