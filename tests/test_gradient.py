"""
Test gradient of least-square cost function
"""
import spin
import numpy as np
import unittest

from scipy.optimize import approx_fprime
from littlehelpers import numerical_gradient, compare_grad, random_walk

class TestGradient(unittest.TestCase):

    def test_gradient(self):

        coords = random_walk(1e3)

        rot = spin.Rotation(spin.random_rotation())

        score = spin.LeastSquares(coords, rot(coords))
        score.trafo.check_matrix = False

        self.assertTrue(score(rot.inverse.matrix) < 1e-10)

        R    = spin.random_rotation()
        grad = score.gradient(R).flatten()

        out = 'eps={0:.0e}: {1:.2e}, {2:.2f} (grad vs num), {3:.2e}, {4:.2f} (grad vs scipy)'

        for eps in np.logspace(-3, -10, 8):

            num  = numerical_gradient(score, R, eps).flatten()
            num2 = approx_fprime(R.flatten(), score, eps)
            args = (eps,) + compare_grad(grad, num) + compare_grad(grad, num2)

            print(out.format(*args))

            err, cc = compare_grad(grad, num)

            self.assertTrue(err < max(eps, 1e-4))
            self.assertTrue(np.fabs(cc - 100) < max(eps, 1e-2))

if __name__ == '__main__':
    unittest.main()
