"""
Testing calculation of nearest rotation matrix.
"""
import spin
import numpy as np
import pylab as plt
import scipy.optimize as opt
import unittest

from csb.bio.utils import fit
from littlehelpers import load_coords, make_title, compare_grad

class TestNearest(unittest.TestCase):

    def test_analytical(self):
        """
        Test analytical calculation of the upper bound using SVD
        and eigenvalue decomposition.
        """
        tol = 1e-10
        n   = int(1e3)

        X, Y = load_coords(['1ake', '4ake'])

        A = np.dot(X.T, Y)
        R = fit(X, Y)[0]

        func  = spin.NearestRotation(A)
        func2 = spin.NearestUnitQuaternion(A)

        self.assertTrue(spin.distance(func.optimum(), R) < tol)
        self.assertTrue(np.linalg.norm(func2.optimum().dofs-spin.Quaternion(R).dofs) < tol)

        rotations = spin.random_rotation(n)

        vals  = np.array(list(map(func, rotations)))
        vals2 = np.dot(rotations.reshape(n,-1), A.flatten())

        self.assertTrue(np.fabs(vals-vals2).max() < tol)
        self.assertTrue(np.all(vals <= func(func.optimum().matrix)))
        self.assertTrue(np.all(vals <= func2(func2.optimum())))

    def test_opt(self):
        """
        Constrained optimization to determine the best unit quaternion
        """
        coords = load_coords(['1ake', '4ake'])

        A = np.dot(coords[0].T,coords[1])
        R = fit(*coords)[0]

        func   = spin.NearestUnitQuaternion(A)
        q_opt  = func.optimum().dofs
        q_opt2 = spin.NearestRotation(A, spin.Quaternion()).optimum().dofs
        
        ## constrained optimization

        constraint = [{'type': 'eq', 'fun': lambda q : np.dot(q,q) - 1}]

        best = -1e308, None

        for n_trials in range(10):

            q_start = spin.Quaternion.random()        
            result  = opt.minimize(lambda q: -func(q), q_start, constraints=constraint)
            q_best  = result['x'] * np.sign(result['x'][0])
            if abs(constraint[0]['fun'](q_best)) < 1e-10 and func(q_best) > best[0]:
                best = func(q_best), q_best

        _, q_best = best

        print(make_title('finding nearest rotation matrix / unit quaternion'))
        print(np.round(q_opt, 5))
        print(np.round(q_best, 5))
        print(np.round(q_opt2, 5))

        tol = 1e-5
        self.assertTrue(np.linalg.norm(q_opt - q_best) < tol)
        self.assertTrue(np.linalg.norm(q_opt - q_opt2) < tol)

    def test_gradients(self):

        print(make_title('testing gradients'))

        A = np.random.standard_normal((3,3))

        func = [spin.NearestRotation(A, trafo=rot()) for rot in \
                (spin.EulerAngles, spin.ExponentialMap,
                 spin.AxisAngle, spin.Quaternion)]
        func+= [spin.NearestUnitQuaternion(A), spin.NearestQuaternion(A)]

        out   = '{0:>14}: err={1:.3e}, cc={2:.2f}'
        grads = []
        
        for f in func:

            x = f.trafo.dofs
            a = f.gradient(x)
            b = opt.approx_fprime(x, f, 1e-7)

            err, cc = compare_grad(a,b)

            print(out.format(f.trafo.name, err, cc))

            self.assertAlmostEqual(err, 0., delta=1e-5)
            self.assertAlmostEqual(cc, 100., delta=1e-2)

            grads.append(a)

    def test_gradients2(self):
        """
        Compare gradients of both implementations using quaternions
        """
        print(make_title('gradient of quaternion-based implementations'))
        
        out   = '{0:>20}: err={1:.3e}, cc={2:.2f}'

        A = np.random.standard_normal((3,3))
        f = spin.NearestRotation(A, trafo=spin.Quaternion())
        g = spin.NearestQuaternion(A)
        h = spin.NearestUnitQuaternion(A)
        
        q = f.trafo.random() if False else np.random.standard_normal(4)
        a = f.gradient(q)
        b = g.gradient(q)
        c = h.gradient(q)
        
        print('f(q)={0:.2e}, g(q)={1:.2e}, h(q)={3:.2e}, norm={2:.2e}'.format(
            f(q), g(q), np.dot(q,q), h(q)))
        print(np.corrcoef(a,b)[0,1])

        print('{0:>22}: {1}'.format(f.__class__.__name__, np.round(a,3)))
        print('{0:>22}: {1}'.format(g.__class__.__name__, np.round(b,3)))
        print('{0:>22}: {1}'.format(h.__class__.__name__, np.round(c,3)))
        
        self.assertAlmostEqual(np.linalg.norm(a-b), 0., delta=1e-5)

if __name__ == '__main__':
    unittest.main()

