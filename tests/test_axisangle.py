"""
Testing the axis-angle parameterization
"""
import spin
import numpy as np
import unittest
import csb.numeric as csb
import matplotlib.pylab as plt

from params import AxisAngle

from littlehelpers import make_title

class TestAxisAngle(unittest.TestCase):
    
    tol = 1e-10

    def test_rotation(self):

        R = spin.random_rotation()

        rot  = spin.AxisAngle.from_rotation(R)
        rot2 = AxisAngle(rot.dofs)

        print(make_title('checking rotations'))

        print('distance Cython vs Python: {0:.3f}'.format(spin.distance(rot, rot2)))
        print('distance Cython vs random: {0:.3f}'.format(spin.distance(rot, R)))
        print('distance Python vs random: {0:.3f}'.format(spin.distance(rot2, R)))

        self.assertTrue(spin.distance(rot, rot2) < self.tol)
        self.assertTrue(spin.distance(rot, R) < self.tol)
        self.assertTrue(spin.distance(rot2, R) < self.tol)

    def test_params(self):

        R = spin.random_rotation()

        rot  = spin.AxisAngle.from_rotation(R)
        rot2 = AxisAngle(rot.dofs)

        axis,   angle = rot.axis_angle
        axis2, angle2 = rot2.axis_angle

        print(make_title('checking axis and angle'))

        print('axis Cython: {}'.format(axis))
        print('axis Python: {}'.format(axis2))
        print('axis csb   : {}'.format(-csb.axis_and_angle(R)[0]))

        print('\nangle Cython: {0:.3f} rad'.format(angle))
        print('angle Python: {0:.3f} rad'.format(angle2))
        print('angle csb   : {0:.3f} rad\n'.format(csb.axis_and_angle(R)[1]))

        self.assertTrue(np.linalg.norm(axis-axis2) < self.tol)
        self.assertTrue(np.fabs(angle-angle2) < self.tol)

        ## comparison to matrix constructed with csb
        
        self.assertTrue(spin.distance(csb.rotation_matrix(-axis, angle), rot2) < self.tol)

    def test_visual(self):
        """
        Test random sampling by visual comparison with true pdf
        """
        n    = int(1e5)
        R    = spin.random_rotation(n)
        dofs = np.array([spin.AxisAngle.from_rotation(r).dofs for r in R]).T

        names   = ('theta', 'phi', 'alpha')
        pdfs    = (spin.Polar, spin.Azimuth, spin.RotationAngle)

        kw_hist = dict(density=True, alpha=0.2, color='k', bins=100)
        kw_plot = dict(alpha=0.7, color='r', lw=3)

        print(make_title('checking random sampling'))

        fig, axes = plt.subplots(1,3,figsize=(12,4))

        for name, pdf, values, ax in zip(names, pdfs, dofs, axes):

            x = pdf.axis(200) - (np.pi if name == 'phi' else 0.)

            p, bins = ax.hist(values, label=r'$\{0}$'.format(name), **kw_hist)[:2]
            ax.plot(x, pdf.prob(x), **kw_plot)        
            ax.legend()

            x = 0.5 * (bins[1:] + bins[:-1])

            mse = np.mean((p - pdf.prob(x))**2)

            print('MSE {0:>5}: {1:.3e}'.format(name, mse) )

            self.assertTrue(mse < 1e-3)

        fig.tight_layout()

if __name__ == '__main__':
    unittest.main()
    

