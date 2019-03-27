"""
Rigid superposition of two protein structures using
non-linear optimization compared to algebraic solution
"""
import spin
import numpy as np
import unittest
import scipy.optimize as opt

from csb.bio.utils import rmsd, fit, fit_wellordered

from littlehelpers import load_coords, compare_grad, make_title

class TestRigid(unittest.TestCase):

    def setUp(self):

        rot_types = (spin.Rotation,
                     spin.EulerAngles,
                     spin.ExponentialMap,
                     spin.AxisAngle,
                     spin.Quaternion)
        
        R = spin.random_rotation()
        t = np.random.standard_normal(3) * 10

        trafos = [spin.RigidMotion(R, t, rot_type) for rot_type in rot_types]

        self.coords = load_coords(['1ake','4ake'])
        self.trafos = trafos

    def test_trafo(self):

        pass
        
    def test_params(self):

        rigid = self.trafos[0]

        print(make_title('comparing matrix and vector'))
        out = '{0:14s} : distance rotation={1:.3e}, translation={2:.3e}'

        for other in self.trafos[1:]:

            dist_rot = spin.distance(rigid.rotation, other.rotation)
            dist_trans = np.linalg.norm(rigid.translation.vector-other.translation.vector)

            print(out.format(other.rotation.name, dist_rot, dist_trans))

            self.assertAlmostEqual(dist_rot, 0., delta=1e-10)
            self.assertAlmostEqual(dist_trans, 0., delta=1e-10)
                             
    def test_rmsd(self):

        R, t = fit(*self.coords)
        out  = '{0:.2f} ({1})'

        print(make_title('RMSD'))
        print(out.format(rmsd(*self.coords), 'SVD'))

        rmsds = []

        for trafo in self.trafos[1:]:

            trafo.matrix_vector = R, t

            r = np.mean(np.sum((self.coords[0] - trafo(self.coords[1]))**2,1))**0.5

            print(out.format(r, trafo.rotation.name))

            rmsds.append(r)

        self.assertAlmostEqual(np.std(rmsds), 0., delta=1e-10)

    def test_grad(self):

        print(make_title('checking gradient'))

        pose = spin.random_rotation(), np.random.standard_normal(3) * 10
        out  = '{0:>14}: corr={1:.1f}, rel.error={2:.3e}'

        for trafo in self.trafos[1:]:
            
            trafo.matrix_vector = pose
            
            f = spin.LeastSquares(*self.coords, trafo=trafo)
            x = trafo.dofs.copy()
            a = f.gradient(x)
            b = opt.approx_fprime(x, f, 1e-8)

            err, cc = compare_grad(a, b)
            
            print(out.format(trafo.rotation.name, cc, err))

            self.assertAlmostEqual(cc, 100., delta=1e-2)
            self.assertAlmostEqual(err, 0., delta=1e-5)

    def test_opt(self):

        print(make_title('test optimization of least-squares residual'))

        out = '{0:>14}: #steps={1:3d}, RMSD: {5:.2f}->{2:.2f}, ' + \
              'accuracy: {3:.3e} (rot), {4:.3e} (trans)'

        start = spin.random_rotation(), np.random.standard_normal(3) * 10
        R, t  = fit(*self.coords)

        rot   = []
        trans = []
        rmsds = []

        for trafo in self.trafos[1:]:

            trafo.matrix_vector = start
            
            f = spin.LeastSquares(*self.coords, trafo=trafo)
            x = trafo.dofs.copy()
            y = opt.fmin_bfgs(f, x, f.gradient, disp=False)

            rot.append(spin.distance(R, trafo.rotation))
            trans.append(np.linalg.norm(t-trafo.translation.vector))
            rmsds.append(np.sqrt(2 * f(y) / len(self.coords[0])))
            
            print(out.format(trafo.rotation.name, len(f.values),
                             rmsds[-1], rot[-1], trans[-1],
                             np.sqrt(2 * f(x) / len(self.coords[0]))))

            self.assertAlmostEqual(rot[-1], 0., delta=1e-5)
            self.assertAlmostEqual(trans[-1], 0., delta=1e-5)

        self.assertAlmostEqual(np.std(rmsds), 0., delta=1e-5)
            
        ## R, t = fit_wellordered(*coords)
        ## trafo.matrix_vector = R, t

        ## print trafo.rotation

        ## trafos = [spin.RigidMotion(spin.Quaternion(spin.Quaternion.random()).matrix,
        ##                            np.random.standard_normal(3)) for _ in range(2)]

        ## coords = np.random.random((10,3))

        ## trafo = trafos[1](trafos[0])

        ## print np.fabs(trafo(coords) - trafos[1](trafos[0](coords))).max()

if __name__ == '__main__':

    unittest.main()
