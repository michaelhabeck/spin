"""
Testing the exponential map
"""
import spin
import numpy as np
import unittest
import csb.numeric as csb
import matplotlib.pylab as plt

from params import ExponentialMap

from scipy.linalg import logm
from spin.rotation import skew_matrix

from littlehelpers import make_title

class TestExpMap(unittest.TestCase):

    tol = 1e-10

    def test_matrix(self):
        
        dofs = spin.ExponentialMap.random()
        rot  = spin.ExponentialMap(dofs)
        rot2 = spin.ExponentialMap.from_rotation(rot)
        rot3 = ExponentialMap(dofs)

        print(make_title('Cython (from dofs)'))
        print(rot)
        
        print(make_title('Cython (from matrix)'))
        print(rot2)

        print(make_title('Python (from dofs)'))
        print(rot3)

        axis, angle   = rot.axis_angle
        r, theta, phi = csb.polar3d(axis)
        axisangle     = spin.AxisAngle([theta, phi, angle])
        rot4          = spin.Rotation(csb.rotation_matrix(axis, -angle))

        self.assertTrue(spin.distance(rot, rot2) < self.tol)
        self.assertTrue(spin.distance(rot, rot4) < self.tol)
        self.assertTrue(spin.distance(rot3, rot4) < self.tol)
        self.assertTrue(spin.distance(rot2, axisangle) < self.tol)

    def test_params(self):
        """
        Compute skewmatrix and dofs
        """
        rot = spin.ExponentialMap()
        R = rot.matrix
        B = logm(R).real

        print(np.round(B, 3))

        axis, theta = rot.axis_angle

        b = axis * theta
        C = 0.5 * (R - R.T) * theta / np.sin(theta)
        c = np.array([-C[1,2], C[0,2], -C[0,1]])
        c = 0.5 * theta / np.sin(theta) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

        self.assertTrue(np.fabs(np.linalg.norm(axis)-1) < self.tol)
        self.assertTrue(np.fabs(B - skew_matrix(b)).max() < self.tol)
        self.assertTrue(np.fabs(B - C).max() < self.tol)
        self.assertTrue(np.linalg.norm(c-b) < self.tol)
        self.assertTrue(np.linalg.norm(c-spin.ExponentialMap.from_rotation(R).dofs) < self.tol)

    def test_random(self):

        n    = int(1e5)
        R    = spin.random_rotation(n)
        dofs = np.array([spin.ExponentialMap.from_rotation(r).dofs for r in R]).T

        names = ('x','y','z')

        fig, axes = plt.subplots(1,4,figsize=(16,4))

        kw_hist = dict(density=True, alpha=0.2, color='k', bins=50)
        kw_plot = dict(alpha=0.7, color='r', lw=3)

        for name, values, ax in zip(names, dofs, axes):
            ax.hist(values,label=r'${0}$'.format(name), **kw_hist)
            ax.legend()

        x = spin.RotationAngle.axis(200)

        ax = axes[-1]
        ax.hist(np.linalg.norm(dofs,axis=0), **kw_hist)
        ax.plot(x, spin.RotationAngle.prob(x), **kw_plot)

        fig.tight_layout()

if __name__ == '__main__':
    unittest.main()

