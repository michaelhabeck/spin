"""
Three-dimensional rotation implemented as subclasses of Transformation. Support
for various parameterizations including quaternions, Euler angles, axis-angle
and the exponential map.
"""
import abc
import numpy as np
import csb.numeric as csb

from .trafo import Transformation

from . import euler
from . import expmap
from . import axisangle
from . import quaternion

from csb.statistics.rand import random_rotation

from scipy.optimize import brentq


def det3x3(a):    
    return +a[0,0] * (a[1,1] * a[2,2] - a[2,1] * a[1,2]) \
           -a[1,0] * (a[0,1] * a[2,2] - a[2,1] * a[0,2]) \
           +a[2,0] * (a[0,1] * a[1,2] - a[1,1] * a[0,2])


def is_rotation_matrix(R):
    """Checks if numpy array is a three-dimensional rotation matrix."""
    return R.shape == (3, 3) and np.isclose(det3x3(R), 1.0)


def skew_matrix(a):
    """Skew-symmetric matrix generated from a 3D vector. Multiplication with
    this matrix with another vector is the same as the cross-product between
    the two vectors.
    """
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


def distance(a, b):
    """Frobenius distance between two three-dimensional rotation matrices. """
    if isinstance(a, Rotation):
        a = a.matrix
    if isinstance(b, Rotation):
        b = b.matrix
    a = a.reshape(-1, 9)
    b = b.reshape(-1, 9)
    return np.squeeze(1 - np.sum(a * b, axis=1) / 3)


def map_to_quat(A):
    """Construct a 4x4 matrix 'M' such that the (linear) inner product between
    the 3x3 matrix 'A' and a rotation 'R' (i.e. 'sum(A*R)') can be written in
    quadratic form: 'np.dot(q, M.dot(q))' where 'q' is the unit quaternion
    encoding 'R'.

    We have:

        M[0, 0] =  A[0, 0] + A[1, 1] + A[2, 2]
        M[1, 1] =  A[0, 0] - A[1, 1] - A[2, 2]
        M[2, 2] = -A[0, 0] + A[1, 1] - A[2, 2]
        M[3, 3] = -A[0, 0] - A[1, 1] + A[2, 2]

        M[0, 1] = M[1, 0] = -A[1, 2] + A[2, 1]
        M[0, 2] = M[2, 0] =  A[0, 2] - A[2, 0] 
        M[0, 3] = M[3, 0] = -A[0, 1] + A[1, 0]

        M[1, 2] = M[2, 1] =  A[0, 1] + A[1, 0]
        M[1, 3] = M[3, 1] =  A[0, 2] + A[2, 0]

        M[2, 3] = M[3, 2] =  A[1, 2] + A[2, 1]    

    Parameters
    ----------
    A : ndarray
        (3, 3) numpy array

    Returns
    -------
    (4, 4) symmetric numpy array
    """
    assert np.shape(A) == (3, 3), '(3, 3) matrix required'
    
    M = np.empty((4, 4), dtype=A.dtype)

    quaternion.map_to_quat(A, M)

    return M


class Angle(object):

    def random(cls, n):
        pass 

    def axis(cls, n):
        pass 

    def log_prob(cls, x):
        pass 

    @classmethod
    def prob(cls, x, normed=True):

        prob = np.exp(cls.log_prob(x))
        if np.iterable(x) and normed:
            prob /= csb.trapezoidal(x, prob)

        return prob

    
class Azimuth(Angle):

    @classmethod
    def random(cls, n):
        """Generate random azimuth angles, ie. uniformly distributed angles.
        """        
        return np.random.random(n) * 2 * np.pi

    @classmethod
    def log_prob(cls, x):
        return 0. * x

    @classmethod
    def axis(cls, n):
        return np.linspace(0., 2 * np.pi, int(n))

    
class Polar(Angle):

    @classmethod
    def random(cls, n=None):
        """Generate random polar angles. """
        return np.arccos(np.random.uniform(-1, 1, size=n))

    @classmethod
    def log_prob(cls, x):
        return csb.log(np.sin(x))

    @classmethod
    def axis(cls, n):
        return np.linspace(0., np.pi, int(n))

    
class RotationAngle(Angle):

    @classmethod
    def random(cls, n=None):
        """Generate random rotation angles, ie. angles following
        \alpha ~ sin^2(\alpha/2)
        """
        if n is None:
            u = np.random.random() * np.pi
            f = lambda x, y=u : x - np.sin(x) - y

            return brentq(f, *cls.axis(2))

        return np.array([cls.random() for _ in range(int(n))])

    @classmethod
    def log_prob(cls, x):
        return csb.log(0.5 * (1 - np.cos(x)))

    @classmethod
    def axis(cls, n):
        return np.linspace(0., np.pi, int(n))

    
class Rotation(Transformation):
    """Rotation

    Three-dimensional rotation matrix.
    """
    def __init__(self, R=np.eye(3)):
        """
        Initialize rotation with a three-dimensional rotation matrix.
        """
        self.check_matrix = True
        
        self.matrix = np.eye(3)
        self.dofs = R

    def _compose(self, other):
        return self.__class__(np.dot(self.matrix, other.matrix))

    def _apply(self, other):
        if other.ndim == 1:
            return np.dot(self.matrix, other)
        else:
            return np.dot(other, self.matrix.T)

    def _invert(self):
        return self.matrix.T

    def map_forces(self, coords, forces):
        """Map Cartesian gradient into space of rotation matrices. """
        return np.dot(forces.T, coords)
    
    def __str__(self):
        return '{0}:\n  {1}'.format(
            self.name, str(np.round(self.matrix, 3)).replace('\n', '\n  '))

    @classmethod
    def random(cls, n=None):
        """Random rotation matrix. """
        if n is None:
            return random_rotation(np.zeros((3, 3)))
        return np.array([cls.random() for _ in range(n)])

    @property
    def dofs(self):
        return self.matrix.flatten()

    @dofs.setter
    def dofs(self, values):

        R = np.reshape(values, (3,3))

        if self.check_matrix and not is_rotation_matrix(R):
            msg = 'Expected a rotation matrix'
            raise ValueError(msg)
                
        self.matrix[...] = R

    def dot(self, A):
        """Returns trace(A.T*R) where R is the rotation matrix. """
        return np.sum(A * self.matrix)

    
class Parameterization(Rotation):
    """Parameterization

    Parameterization of a three-dimensional rotation matrix.
    """    
    @classmethod
    def _from_matrix(cls, R):
        raise NotImplementedError

    def _to_matrix(self):
        raise NotImplementedError

    @property
    def jacobian(self):
        raise NotImplementedError

    @property
    def matrix(self):
        return self._to_matrix()

    @matrix.setter
    def matrix(self, R):
        if not is_rotation_matrix(R):
            msg = 'Input matrix must be a rotation matrix'
            raise ValueError(msg)                        
        self.dofs = self.__class__._from_matrix(R)

    @property
    def dofs(self):
        return self._dofs

    @dofs.setter
    def dofs(self, dofs):
        self._dofs[...] = dofs

    def __init__(self, dofs=None):

        self._dofs = self.__class__._from_matrix(np.eye(3))
        
        if dofs is None:
            dofs = self.__class__.random()
        elif np.iterable(dofs):
            dofs = np.array(dofs)
        else:
            msg = 'Argument must be None or iterable'
            raise TypeError(msg)
        
        if dofs.ndim == 1:
            self.dofs = dofs
        elif dofs.ndim == 2:
            self.matrix = dofs
        else:
            msg = 'Argument must be DOF vector or rotation matrix'
            raise ValueError(msg)

    @classmethod
    def from_rotation(cls, rotation):
        """Calculate parameters from rotation matrix. """
        if isinstance(rotation, Rotation):
            R = rotation.matrix
        elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            R = rotation
        else:
            msg = 'Argument must be instance of Rotation or 3x3 numpy array'
            raise TypeError(msg)
        
        return cls(cls._from_matrix(R))

    def map_forces(self, coords, forces):
        """Map Cartesian gradient onto parameter space by means of the chain
        rule. 
        """
        grad = super().map_forces(coords, forces).flatten()
        return np.sum(self.jacobian.reshape(self.n_dofs,- 1) * grad, axis=1)

    
class EulerAngles(Parameterization):
    """EulerAngles

    Cython implementation.
    """
    @classmethod
    def _from_matrix(cls, R):
        return np.array(euler.params(np.ascontiguousarray(R)))

    def _to_matrix(self):
        R = np.ascontiguousarray(np.zeros((3,3)))
        euler.matrix(self.dofs, R)
        return R

    @property
    def jacobian(self):
        J = np.zeros((3,3,3))
        euler.jacobian(self.dofs, J[0], J[1], J[2])
        return J

    @classmethod
    def random(cls, n=None):
        """Generate random Euler angles. """
        return np.array([Azimuth.random(n), Polar.random(n), Azimuth.random(n)])
    

class AxisAngle(Parameterization):

    @property
    def axis_angle(self):
        theta, phi, angle = self.dofs
        axis = np.zeros(3)
        axisangle.axis(theta, phi, axis)
        
        return axis, angle

    @classmethod
    def _from_matrix(cls, R):
        return np.array(axisangle.params(np.ascontiguousarray(R)))

    def _to_matrix(self):
        R = np.ascontiguousarray(np.zeros((3, 3)))
        axisangle.matrix(self.dofs, R)
        return R

    @property
    def jacobian(self):
        J = np.zeros((3, 3, 3))
        axisangle.jacobian(self.dofs, *J)
        return J

    @classmethod
    def random(cls, n=None):
        """Generate random axis and angle. """
        
        return np.array(
            [Polar.random(n), Azimuth.random(n), RotationAngle.random(n)])
    

class ExponentialMap(AxisAngle):
    """ExponentialMap

    Parameterization of rotations in terms of the exponential map. 
    """
    @property
    def axis_angle(self):
        a = np.linalg.norm(self.dofs)
        n = self.dofs / a

        return n, np.mod(a, 2*np.pi)

    @classmethod
    def _from_matrix(cls, R):
        return np.array(expmap.params(np.ascontiguousarray(R)))
        
    def _to_matrix(self):
        R = np.ascontiguousarray(np.zeros((3, 3)))
        expmap.matrix(self.dofs, R)
        return R

    @property
    def jacobian(self):
        norm = np.linalg.norm(self.dofs)**2 + 1e-100
        v = self.dofs

        A = skew_matrix(v)
        R = self.matrix
        B = np.cross(v, R - np.eye(3))

        return np.array([np.dot(v[i] * A + skew_matrix(B[:,i]), R)
                         for i in range(3)]) / norm

    @classmethod
    def random(cls, n=None):
        a = RotationAngle.random(n)
        if n is None:
            u = np.random.standard_normal(3)
            u/= np.sum(u**2)**0.5
        else:
            u = np.random.standard_normal((3,n))
            u/= np.sum(u**2,0)**0.5

        return u * a
        
    def rotate(self, v):
        """Rodrigues formula. """

        n, a = self.axis_angle

        return np.cos(a) * v \
             + np.sin(a) * np.cross(n, v) \
             + (1-np.cos(a)) * np.dot(v, n) * n

    
class Quaternion(Parameterization):
    """Quaternion

    Parameterization of a three-dimensional rotation matrix in terms of a unit
    quaternion.
    """
    def _to_matrix(self):
        R = np.ascontiguousarray(np.zeros((3, 3)))
        quaternion.matrix(self.dofs, R)
        return R

    @classmethod
    def _from_matrix(cls, R):
        return np.array(quaternion.params(np.ascontiguousarray(R)))

    @property
    def jacobian(self):
        J = np.zeros((4, 3, 3))
        quaternion.jacobian(self.dofs, *J)
        return J

    @classmethod
    def from_axis_angle(cls, axis, angle):
        return np.append(np.cos(angle*0.5), axis * np.sin(angle*0.5))

    @classmethod
    def random(cls, n=None, upper_sphere=True):

        if n is None:
            q = np.random.randn(4)
            q/= np.linalg.norm(q)

        else:
            q = np.random.standard_normal((4,n))
            q/= np.linalg.norm(q,axis=0)

        if upper_sphere:
            q[0] = np.fabs(q[0])

        return q
            
    @property
    def axis_angle(self):

        self.normalize()

        q = self.dofs
        
        theta = 2 * np.arctan2(np.linalg.norm(q[1:]), q[0])

        if np.isclose(theta, 0):
            axis = np.array([0, 0, 1])
        else:
            axis = q[1:] / np.sqrt(1-q[0]**2)

        return theta, axis

    def normalize(self):
        self.dofs /= np.linalg.norm(self.dofs) + 1e-100

    def rotate(self, v):
        """Rotate 3d vector. """
        x = 2 * np.cross(self.dofs[1::], v)

        return v + self.dofs[0] * x + np.cross(self.dofs[1::], x)

def random_rotation(n=None, rotation_type=EulerAngles):

    dofs = rotation_type.random(n)
    if dofs.ndim == 1:
        return rotation_type(dofs).matrix
    else:
        return np.array([rotation_type(x).matrix for x in dofs.T])

