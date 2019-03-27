"""
Rigid transformation
"""
import numpy as np

from .trafo import Transformation, Translation
from .rotation import Rotation

class RigidMotion(Transformation):
    """RigidMotion

    Rotation followed by a translation.
    """
    @property
    def matrix_vector(self):
        return self.rotation.matrix, self.translation.vector

    @matrix_vector.setter
    def matrix_vector(self, trafo):
        R, t = trafo
        self.rotation.matrix = R
        self.translation.vector[...] = t

    @property
    def inverse(self):
        R, t = self._invert()
        return self.__class__(R, t, self.rotation.__class__)

    @property
    def dofs(self):
        return np.append(self.rotation.dofs, self.translation.dofs)

    @dofs.setter
    def dofs(self, dofs):

        m = self.rotation.n_dofs
        n = self.translation.n_dofs

        self.rotation.dofs    = dofs[:m]
        self.translation.dofs = dofs[m:m+n]

    def __init__(self, R=np.eye(3), t=np.zeros(3), rotation_type=Rotation):
        """
        Parameters
        ----------
        R : numpy array
          Three-dimensional rotation matrix

        t : numpy array
          Three-dimensional translation vector

        rotation_type : subclass of Rotation
          Optional parameterization 
        """
        if not issubclass(rotation_type, Rotation):
            msg = 'rotation type must be subclass of Rotation'
            raise TypeError(msg)
        
        self.rotation    = rotation_type(R)
        self.translation = Translation(t)

    def _apply(self, other):
        return self.translation._apply(self.rotation._apply(other))

    def _invert(self):
        R, t = self.matrix_vector
        return R.T, -np.dot(R.T, t)

    def _compose(self, other):

        R = self.rotation(other.rotation).matrix
        t = self.translation(self.rotation(other.translation.vector))

        return self.__class__(R, t, rotation_type=self.rotation.__class__)

    def map_forces(self, coords, forces):

        grad_rot   = self.rotation.map_forces(coords, forces).flatten()
        grad_trans = self.translation.map_forces(coords, forces)

        return np.append(grad_rot, grad_trans)

