import spin
import numpy as np
import pylab as plt
import scipy.optimize as opt
import matplotlib.pylab as plt

from littlehelpers import load_coords

coords = load_coords(['1ake', '4ake'])

objectives= [spin.LeastSquares(*coords, trafo=rot()) for rot in
             [spin.EulerAngles, spin.ExponentialMap, spin.AxisAngle]]

rotations = spin.random_rotation(1000)

results = []

for R in rotations:

    results.append([])

    for func in objectives:

        start = func.trafo.from_rotation(R).dofs
        func.values = []
        end   = opt.fmin_bfgs(func, start, fprime=func.gradient, disp=False)

        results[-1] += [func(end), len(func.values)]

results = np.array(results)    

names = [func.trafo.name for func in objectives]

print(np.round(results[:,1::2].mean(0), 1))
print(np.var(results[:,::2],0))

fig, ax = plt.subplots(1,2,figsize=(10,5))
for k, name in enumerate(names):
    ax[0].hist(results[:,2*k+1],bins=30,alpha=0.5,density=True,label=name)
ax[0].legend()
for k, name in enumerate(names):
    ax[1].hist(results[:,2*k],bins=30,alpha=0.5,density=True,label=name)
ax[1].legend()
fig.tight_layout()
