#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: PLOT.PY
Date: Friday, April  5 2013
Description: Evaluate pose experiment.
"""

import cPickle as pickle
import pylab
import matplotlib as mpl

y = [0.0]
p = []
for i in range(4):
    (avg_reward,avg_length,p_avg_reward,p_avg_length) = pickle.load(open('pose_reward{0}.pck'.format(i)))
    print (avg_reward,avg_length,p_avg_reward,p_avg_length)
    y.append(avg_reward)

p = [p_avg_reward] * len(y)
x = range(len(y))

pylab.clf()
pylab.plot(x,y,label="Learned")
pylab.plot(x,p,label="Optimal")
pylab.title('Alignment Performance with Sensorimotor Embedding')

pylab.ylim([0.0,1.1])

pylab.ylabel('Average Reward')
pylab.xlabel('# of Policy Iterations')
pylab.legend(loc=4)
pylab.gca().set_xticks(range(5))
#pylab.show()
pylab.savefig('alignment.pdf')
