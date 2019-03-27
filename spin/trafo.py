"""
Affine transformation of three-dimensional coordinates
"""
import numpy as np

from functools import reduce

def compose(*trafos):
    """
    Compose a sequence a transformations.
    """
    return reduce(lambda a, b: a(b), trafos)

class Transformation(object):
    """Transformation

    Abstract class representating a transformation of a set of
    coordinates. 
    """
    ## these methods / properties need to be implemented by classes
    ## inherited from Transformation
    
    ## >>>>>>>>>

    def _compose(self, other):
        raise NotImplementedError

    def _apply(self, other):
        raise NotImplementedError

    def _invert(self):
        raise NotImplementedError

    def map_forces(self, coords, forces):
        """
        Map Cartesian forces into space of transformation parameters.
        """
        raise NotImplementedError
    
    @property
    def dofs(self):
        raise NotImplementedError

    ### <<<<<<<<

    def __call__(self, other):
        """
        Apply transformation to another transformation, vector or
        matrix of vectors.
        """
        if isinstance(other, self.__class__):
            return self._compose(other)
        elif type(other) == np.ndarray and other.ndim in (1,2):
            return self._apply(other)
        else:
            msg = 'Argument must be other transformation or vector or matrix'
            return TypeError(msg)

    @property
    def inverse(self):
        return self.__class__(self._invert())

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def n_dofs(self):
        return len(self.dofs)
    
class Translation(Transformation):
    """Translation

    Shift by a constant vector
    """
    def __init__(self, translation):
        self.vector = np.array(translation)
        assert self.vector.ndim == 1
        
    def _compose(self, other):
        return self.__class__(self.vector + other.vector)

    def _apply(self, other):
        return other + self.vector

    def _invert(self):
        return -self.vector

    def map_forces(self, coords, forces):
        return np.sum(forces,0)

    @property
    def dofs(self):
        return self.vector

    @dofs.setter
    def dofs(self, values):
        self.vector[...] = values
