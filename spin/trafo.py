"""
Affine transformation of three-dimensional coordinates
"""
import abc
import numpy as np

from functools import reduce


def compose(*trafos):
    """Compose a sequence a transformations."""
    return reduce(lambda a, b: a(b), trafos)


def is_vector(array):
    """True if argument is ndarray of rank 1. """
    return isinstance(array, np.ndarray) and array.ndim == 1


def is_matrix(array):
    """True if argument is ndarray of rank 2. """
    return isinstance(array, np.ndarray) and array.ndim == 2


class Transformation(object):
    """Transformation

    Abstract class representing a transformation of a set of coordinates. 
    """
    def _compose(self, other):
        pass

    def _apply(self, other):
        pass 

    def _invert(self):
        pass

    def map_forces(self, coords, forces):
        """Map Cartesian forces into space of transformation parameters. """
        pass
    
    def dofs(self):
        raise NotImplementedError

    def __call__(self, other):
        """Applies transformation to another object.

        Apply transformation to another object which could be another
        transformation, a vector or a row-based array of vectors.
        """
        if isinstance(other, self.__class__):
            return self._compose(other)
        elif is_vector(other) or is_matrix(other):
            return self._apply(other)
        else:
            msg = 'Expected another transformation, a vector or a matrix'
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

    Shift by a constant vector. 
    """
    def __init__(self, translation):
        if not np.iterable(translation):
            raise TypeError('Expected an iterable')

        translation = np.array(translation)
        if not is_vector(translation):
            raise TypeError('Expected a vector')

        self.vector = translation
        
    def _compose(self, other):
        return self.__class__(self.vector + other.vector)

    def _apply(self, other):
        return other + self.vector

    def _invert(self):
        return -self.vector

    def map_forces(self, coords, forces):
        return np.sum(forces, axis=0)

    @property
    def dofs(self):
        return self.vector

    @dofs.setter
    def dofs(self, values):
        self.vector[...] = values
