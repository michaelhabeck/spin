import spin
import numpy as np

from scipy.spatial.distance import squareform, cdist, pdist

n    = 1000
rot  = (spin.EulerAngles(), spin.ExponentialMap(), spin.AxisAngle(), spin.Quaternion())[-1]
dofs = rot.random(n).T
R    = []
for x in dofs:
    rot.dofs = np.ascontiguousarray(x)
    R.append(rot.matrix.copy())

R = np.array(R)
a = pdist(dofs)

if isinstance(rot, spin.Quaternion):
    ## see Mitchell et al: Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration
    a = squareform(np.dot(dofs,dofs.T),checks=False)
    a = np.arccos(np.fabs(a))
    
b = [spin.distance(R1,R2) for i, R1 in enumerate(R) for R2 in R[i+1:]]

figure()
title(rot.__class__.__name__)
scatter(a,b,alpha=0.1,s=1)
