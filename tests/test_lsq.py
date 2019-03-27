"""
Test least squares fitting using different parameterizations
of rotation matrices
"""
import spin
import numpy as np
import pylab as plt
import scipy.optimize as opt
import unittest

from csb.bio.utils import rmsd, fit

from collections import OrderedDict

from littlehelpers import compare_grad, load_coords, make_title

class TestLSQ(unittest.TestCase):
    """TestLSQ

    Testing various strategies to optimize the least-squares residual
    """
    def setUp(self):

        self.coords = coords = load_coords(['1ake', '4ake'])

        self.lsq = dict(svd=spin.LeastSquares(*coords),
                        euler=spin.LeastSquares(*coords, trafo=spin.EulerAngles()),
                        expmap=spin.LeastSquares(*coords, trafo=spin.ExponentialMap()),
                        axisangle=spin.LeastSquares(*coords, trafo=spin.AxisAngle()),
                        quaternion=spin.LeastSquares(*coords, trafo=spin.Quaternion()))

    def test_lsq(self):

        rotation, score = self.lsq['svd'].optimum()

        rmsd_ = [np.sqrt(score/len(self.coords[0])),
                 self.lsq['svd'].rmsd(rotation.matrix),
                 rmsd(*self.coords)]
        lsq_  = [0.5 * score, self.lsq['svd'](rotation.matrix)]

        for name in ('euler', 'axisangle', 'expmap', 'axisangle'):

            dofs = self.lsq[name].trafo.from_rotation(rotation).dofs
            lsq_.append(self.lsq[name](dofs))

        rmsd_ = np.round(rmsd_, 5)
        lsq_  = np.round(lsq_, 2)

        print(make_title('checking LSQ optimization using SVD'))
        print('RMSD: {0}'.format(rmsd_))
        print(' LSQ: {0}'.format(lsq_))

        tol = 1e-10
        
        self.assertTrue(np.all(np.fabs(rmsd_ - rmsd_[0]) < tol))
        self.assertTrue(np.all(np.fabs(lsq_ - lsq_[0]) < tol))
        self.assertAlmostEqual(spin.distance(fit(*self.coords)[0], rotation.matrix), 0., delta=tol)

    def test_gradient(self):
        """
        Numerical vs analytical gradient
        """
        print(make_title('Checking gradient'))

        out = '{0:>15}: rel.error={1:.2e}, corr={2:.2f}%'

        for name, lsq in self.lsq.items():

            if name == 'svd': continue

            dofs = lsq.trafo.random()
            grad = lsq.gradient(dofs)
            num  = opt.approx_fprime(dofs, lsq, 1e-7)

            err, cc = compare_grad(grad, num)

            print(out.format(lsq.trafo.name, err, cc))
            
            self.assertAlmostEqual(cc, 100, delta=1e-2)
            self.assertAlmostEqual(err, 0., delta=1e-5)

    def test_optimizers(self):
        """
        Test various optimizers for finding the optimal rotation
        in Euler parameterization
        """
        optimizers = OrderedDict()
        optimizers['nedler-mead'] = opt.fmin
        optimizers['powell'] = opt.fmin_powell
        optimizers['bfgs'] = opt.fmin_bfgs

        print(make_title('Testing different optimizers'))

        rotation, _ = self.lsq['svd'].optimum()
        lsq_opt = self.lsq['svd'](rotation.matrix)
        output  = '{0:11s} : min.score={1:.2f}, dist={2:.3e}, nsteps={3:d}'

        for name, lsq in self.lsq.items():

            if name == 'svd': continue

            print(make_title(lsq.trafo.name))

            start = lsq.trafo.random()

            results = OrderedDict()

            for method, run in optimizers.items():

                lsq.values = []

                args = [lsq, start.copy()] + ([lsq.gradient] if method=='bfgs' else [])
                best = run(*args, disp=False)

                results[method] = np.array(lsq.values)

                summary = lsq(best), spin.distance(rotation, lsq.trafo), len(lsq.values)
                
                print(output.format(*((method,) + summary)))

            fig, ax = plt.subplots(1,1,figsize=(10,6))
            fig.suptitle(lsq.trafo.name)
            for method, values in results.items():
                ax.plot(values, lw=5, alpha=0.7, label=method)
            ax.axhline(lsq_opt, lw=3, ls='--', color='r')
            ax.legend()

if __name__ == '__main__':

    unittest.main()
    
    if False:

        ## trying to visualize energy landscape

        n = 50
        a = np.linspace(0., 2.*np.pi, n)
        b = np.linspace(0., np.pi, n)
        c = np.linspace(0., 2*np.pi, n)

        f = []
        for aa in a:
            for bb in b:
                for cc in c:
                    f.append(score(np.array([aa,bb,cc])))

        beta = 0.01
        f = np.reshape(f, (n,n,n))
        f = np.sqrt(2 * f / len(score.target))
        f = np.exp(-beta * (f-f.min()))
