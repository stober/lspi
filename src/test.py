import numpy as np
import scipy.sparse as sp

x = np.arange(10)
y = np.arange(10,20)

print np.outer(x,y)

s = sp.dok_matrix((10,1))
t = sp.dok_matrix((10,1))
for i in range(10):
	s[i,0] = i
	t[i,0] = i + 10

r = sp.kron(t,s.T).T
print r.todense()
