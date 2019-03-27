"""
Testing various ways of generating random rotations
"""
import spin
import numpy as np
import csb.numeric as csb
import unittest
import matplotlib.pylab as plt

def p_expmap(x, y=None, z=None):

    if type(x) == int: x = np.linspace(-np.pi, np.pi, x)

    if y is None: y = x
    if z is None: z = x

    r = np.sum(np.array(np.meshgrid(x,y,z))**2,0)

    return (x, y, z), r, (1 - np.cos(np.sqrt(r))) / (r + 1e-300)

class TestRandom(unittest.TestCase):

    def test_random(self):

        n = int(1e5)
        R = spin.random_rotation(n)
        nbins = 200
        
        kw_hist = dict(density=True, alpha=0.2, color='k', bins=50)
        kw_plot = dict(alpha=0.7, color='r', lw=3)
        fig, ax = plt.subplots(4,3,figsize=(12,16))

        ## Euler angles

        names  = 'alpha', 'beta', 'gamma'
        angles = spin.Azimuth, spin.Polar, spin.Azimuth
        dofs   = np.transpose(list(map(spin.EulerAngles._from_matrix, R)))

        for k, (name, angle, values) in enumerate(zip(names, angles, dofs)):

            x = angle.axis(nbins)

            ax[0,k].hist(values, label=r'$\{0}$'.format(name), **kw_hist)
            ax[0,k].plot(x, angle.prob(x), **kw_plot)        

        ## axis-angle

        names  = 'theta', 'phi', 'alpha'
        angles = spin.Polar, spin.Azimuth, spin.RotationAngle
        dofs   = np.transpose(list(map(spin.AxisAngle._from_matrix, R)))

        for k, (name, angle, values) in enumerate(zip(names, angles, dofs)):

            x = angle.axis(nbins) - (np.pi if name == 'phi' else 0.)

            ax[1,k].hist(values,label=r'$\{0}$'.format(name), **kw_hist)
            ax[1,k].plot(x, angle.prob(x), **kw_plot)        

        ## exponential map

        names  = ('x','y','z')
        dofs   = np.transpose(list(map(spin.ExponentialMap._from_matrix, R)))

        (x, _, _), r, y = p_expmap(100)
        m = (r < np.pi**2).astype('i')
        y = (m*y).sum(-1).sum(-1)
        y/= csb.trapezoidal(x, y)

        z = np.cos(0.5 * x)
        z/= csb.trapezoidal(x, z)

        for k, (name, values) in enumerate(zip(name, dofs)):

            ax[2,k].hist(values,label=r'${0}$'.format(name), **kw_hist)
            ax[2,k].plot(x, y, **kw_plot)
            kw_plot['color'] = 'b'
            ax[2,k].plot(x, z, **kw_plot)
            kw_plot['color'] = 'r'        

        ## quaternion

        names = ('x','y','z')
        dofs  = np.transpose(list(map(spin.Quaternion._from_matrix, R))[:3])

        for k, (name, values) in enumerate(zip(names, dofs)):
            ax[3,k].hist(values,label=r'${0}$'.format(name), **kw_hist)        

        for a in ax.flat:
            a.legend()

        fig.tight_layout()

if __name__ == '__main__':
    unittest.main()
