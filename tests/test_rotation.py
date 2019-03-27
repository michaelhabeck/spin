"""
Testing rotation module
"""
import spin
import spin.rotation
import numpy as np
import unittest

from littlehelpers import make_title

class TestRotation(unittest.TestCase):

    def test_skew_matrix(self):

        a, b = np.random.standard_normal((2,3))
        A = spin.rotation.skew_matrix(a)

        x = np.dot(A, b)
        y = np.cross(a, b)

        print(make_title('action of skew matrix vs cross product'))
        print('  Skew matrix: {0}'.format(np.round(x, 5)))
        print('Cross product: {0}'.format(np.round(y, 5)))

        for i in range(len(x)):
            self.assertAlmostEqual(x[i], y[i], delta=1e-10)

    def test_is_rotation(self):

        A = np.random.standard_normal((3,3))

        self.assertFalse(spin.rotation.is_rotation_matrix(A))
        self.assertTrue(spin.rotation.is_rotation_matrix(np.linalg.svd(A)[0]))

    def test_angles(self):

        print(make_title('angular distributions'))

        n = int(1e5)
        nbins = n / 100
        tol = 1e-2
        
        for angle in (spin.Azimuth, spin.Polar, spin.RotationAngle):

            x = angle.random(n)
            p, bins = np.histogram(x, bins=nbins, density=True)
            q = angle.prob(0.5 * (bins[1:] + bins[:-1]))

            mse = np.mean((q-p)**2)

            print('{0:>15}: MSE={1:.3e}'.format(angle.__name__, mse))

            min_, max_ = angle.axis(2)

            self.assertTrue(mse < tol)
            self.assertTrue(min_ <= np.min(x))
            self.assertTrue(max_ >= np.max(x))
            
if __name__ == '__main__':
    unittest.main()
                       
