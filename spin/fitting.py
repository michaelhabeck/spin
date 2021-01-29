"""
Least-squares fitting and nearest rotation matrix
"""
import numpy as np
import scipy.linalg as la

from .trafo import Transformation
from .rotation import Rotation, Quaternion, map_to_quat


def qfit(target, source):
    """Least-squares fitting of source onto target using unit quaternions. 

    Parameters
    ----------
    target : (N, 3) array
        3D point cloud onto which the source will be transformed

    source : (N, 3) array
        3D point cloud that will be transformed so as to fit the target
        optimally in a least-squares sense

    Returns
    -------
    R : (3, 3) array
        Optimal rotation matrix

    t : (3, ) array
        Optimal translation vector
    """
    assert target.ndim == 2
    assert np.shape(target)[1] == 3
    assert np.shape(target) == np.shape(source)
    
    x = target.mean(0)
    y = source.mean(0)
    A = np.dot((target-x).T, source-y)
    M = map_to_quat(A)

    _, q = la.eigh(M, eigvals=[3, 3])

    R = Quaternion(q.flatten()).matrix
    t = x - R.dot(y)

    return R, t


class LeastSquares(object):
    """LeastSquares

    Objective function using a least-squares criterion
    """
    def __init__(self, target, source, trafo=Rotation()):
        """
        Parameters
        ----------
        target, source : rank-2 numpy arrays
          N x 3 coordinate arrays

        trafo : instance of Transformation class
          Optional parameterization of the rotation matrix
        """
        if target.shape != source.shape or target.ndim != 2 \
          or target.shape[1] != 3:
            msg = 'input coordinate arrays must have rank 2 and shape (n,3)'
            raise ValueError(msg)

        if not isinstance(trafo, Transformation):
            msg = 'trafo must be instance of Transformation'
            raise TypeError(msg)
        
        self.target = target
        self.source = source
        self.trafo  = trafo

        self.values = []
        
    def forces(self, params):
        """Displacement vectors between both coordinate arrays after rotation
        of the second array. 
        """
        self.trafo.dofs = params
        return self.trafo(self.source) - self.target

    def __call__(self, dofs):
        """Least-squares residual. 
        """
        residual = 0.5 * np.sum(self.forces(dofs)**2)
        self.values.append(residual)

        return residual

    def gradient(self, dofs):
        """Gradient of least-squares residual with respect to rotational
        parameters. 
        """
        forces = self.forces(dofs)
        coords = self.source

        return self.trafo.map_forces(coords, forces)

    def rmsd(self, dofs):
        """
        Root mean square deviation between coordinate arrays
        after rotation given by rotational parameters        
        """
        return np.sqrt(2 * self(dofs) / len(self.target))

    def optimum(self):
        """
        Optimal rotation minimizing the least-squares residual calculated
        by singular value decomposition
        """
        U, L, V = np.linalg.svd(np.dot(self.target.T, self.source))

        R = np.dot(U, V)

        if np.linalg.det(R) < 0:
            R *= -np.eye(3)[2]
            L[2] *= -1

        rmsd = np.sum(self.target**2) + np.sum(self.source**2) - 2 * L.sum()

        return self.trafo.__class__(R), rmsd

    
class NearestRotation(object):
    """NearestRotation

    Finding the rotation matrix that is closest (in a least-squares sense)
    to some general 3x3 matrix.
    """
    def __init__(self, A, trafo=Rotation()):
        """
        Parameters
        ----------
        A : 3 x 3 array
          Input matrix to which the closest rotation matrix shall be computed

        trafo : instance of Rotation class
          Optional parameterization of the rotation matrix
        """
        if A.shape != (3, 3):
            msg = 'Shape of input matrix must be (3,3)'
            raise ValueError(msg)

        if not isinstance(trafo, Rotation):
            msg = 'trafo must be instance of Rotation'
            raise TypeError(msg)
        
        self.A     = A
        self.trafo = trafo

        self.values = []

    def __call__(self, dofs):
        """
        Inner product between rotation matrix and input target matrix
        """
        self.trafo.dofs = dofs
        return np.sum(self.A * self.trafo.matrix)
    
    def gradient(self, dofs):
        """
        Gradient of inner product with respect to rotational parameters
        """
        self.trafo.dofs = dofs
        if hasattr(self.trafo, 'jacobian'):
            return np.array([np.sum(self.A * J) for J in self.trafo.jacobian])
        else:
            return self.A
        
    def optimum(self):
        """
        Closest rotation matrix determined by singular value decomposition
        """
        U, L, V = np.linalg.svd(self.A)

        R = np.dot(U, V)

        if np.linalg.det(R) < 0:
            R *= -np.eye(3)[2]

        return self.trafo.__class__(R)

    
class NearestUnitQuaternion(NearestRotation):
    """NearestUnitQuaternion

    Finding the rotation matrix (parameterized by a unit quaternion) that is
    closest (in a least-squares sense) to some general 3x3 matrix.
    """
    def __init__(self, A):
        """
        Parameters
        ----------
        A : 3 x 3 array
          Input matrix to which the closest rotation matrix shall be computed
        """
        super(NearestUnitQuaternion, self).__init__(A, Quaternion())

        self.M = map_to_quat(A)

    def __call__(self, q):
        """
        Inner product between rotation matrix and input target matrix
        """
        if isinstance(q, Quaternion):
            q = q.dofs

        return np.dot(q, np.dot(self.M, q))

    def gradient(self, q):
        if isinstance(q, Quaternion):
            q = q.dofs

        return 2 * np.dot(self.M, q)

    def optimum(self):
        """
        Returns quaternion parameterizing closest rotation matrix
        determined by spectral decomposition
        """
        v, U = np.linalg.eigh(self.M)
        q = U[:,v.argmax()]

        return Quaternion(q * np.sign(q[0]))

    
class NearestQuaternion(NearestUnitQuaternion):
    """NearestQuaternion

    Finding the rotation matrix (parameterized by a general quaternion) that is
    closest (in a least-squares sense) to some general 3x3 matrix.
    """
    def __call__(self, q):
        """Inner product between rotation matrix and input target matrix. """
        
        if isinstance(q, Quaternion): q = q.dofs
            
        return super(NearestQuaternion, self).__call__(q) / np.dot(q, q)

    def gradient(self, q):
        """Gradient taking into account that input quaternion does not need to
        lie on the 4d-sphere. 
        """
        if isinstance(q, Quaternion): q = q.dofs

        grad = super(NearestQuaternion, self).gradient(q)
        return (grad - 2*self(q)*q) / np.dot(q, q)


    
