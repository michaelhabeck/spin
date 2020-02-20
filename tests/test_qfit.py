"""
Compare least-squares fitting with unit quaternions (Kearsley's algorithm)
with fitting using singular value decomposition (Kabsch's algorithm). 
"""
from __future__ import print_function

import spin
import time
import numpy as np

from littlehelpers import load_coords

from csb.bio.utils import fit

X, Y = load_coords(['1ake', '4ake'])

n = 1000
X = np.repeat(X,n,axis=0)
Y = np.repeat(Y,n,axis=0)

t = time.clock()
A = spin.qfit(X,Y)
t = time.clock()-t

t2 = time.clock()
B  = fit(X,Y)
t2 = time.clock()-t2

print('dist(R_quat,R_svd) = {0:.3f}'.format(spin.distance(A[0], B[0])))
print('dist(t_quat,t_svd) = {0:.3f}'.format(np.linalg.norm(A[1]-B[1])))
print('times: {0:.3f} vs {1:.3f} (quat vs svd)'.format(t, t2))
