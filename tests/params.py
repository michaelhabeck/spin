"""
Alternative implementations of the rotation matrix parameterizations
using pure Python code rather than the Cython implementation provided
by the spin library.
"""
import spin
import numpy as np
import csb.numeric as csb

from spin.rotation import skew_matrix

class EulerAngles(spin.EulerAngles):
    """EulerAngles
    
    Implementation based on Python functions provided by the csb library.
    """
    @classmethod
    def _from_matrix(cls, R):
        return csb.euler_angles(R)

    def _to_matrix(self):
        return csb.euler(*self.dofs)

class AxisAngle(spin.AxisAngle):
    """AxisAngle

    Pure Python-based implementation using functions from csb.
    """
    @classmethod
    def _from_matrix(cls, R):

        axis, angle   = csb.axis_and_angle(R)
        _, theta, phi = csb.polar3d(axis)

        return np.array([theta, phi, angle])

    @property
    def axis_angle(self):
        theta, phi, angle = self.dofs
        axis = csb.from_polar3d(np.array([1, theta, phi]))

        return axis, angle

    def _to_matrix(self):
        n, a = self.axis_angle
        return np.cos(a) * np.eye(3) + np.sin(a) * skew_matrix(n) + \
               (1-np.cos(a)) * np.multiply.outer(n,n)

class ExponentialMap(spin.ExponentialMap):
    """ExponentialMap

    Pure Python-based implementation of the exponential map
    """
    @classmethod
    def _from_matrix(cls, R):
        
        theta, phi, angle = axisangle.params(R)
        axis = np.zeros(3)
        axisangle.axis(theta, phi, axis)

        return angle * axis

    def _to_matrix(self):

        n, a = self.axis_angle
        
        return np.cos(a) * np.eye(3) + np.sin(a) * skew_matrix(n) + \
               (1-np.cos(a)) * np.multiply.outer(n,n)
        
class Quaternion(spin.Quaternion):

    def _to_matrix(self):

        a, b, c, d = self.dofs / np.linalg.norm(self.dofs)

        return np.array([[a**2+b**2-c**2-d**2, 2*b*c-2*a*d, 2*b*d+2*a*c],
                         [2*b*c+2*a*d, a**2-b**2+c**2-d**2, 2*c*d-2*a*b],
                         [2*b*d-2*a*c, 2*c*d+2*a*b, a**2-b**2-c**2+d**2]])

    @classmethod
    def _from_matrix(cls, R):

        q = np.array([1 + R[0,0] + R[1,1] + R[2,2],
                      1 + R[0,0] - R[1,1] - R[2,2],
                      1 - R[0,0] + R[1,1] - R[2,2],
                      1 - R[0,0] - R[1,1] + R[2,2]])

        i = q.argmax()

        q[i] = 0.5 * q[i]**0.5

        if i == 0:
            
            q[1] = 0.25 * (R[2,1]-R[1,2]) / q[i]
            q[2] = 0.25 * (R[0,2]-R[2,0]) / q[i]
            q[3] = 0.25 * (R[1,0]-R[0,1]) / q[i]

        elif i == 1:

            q[0] = 0.25 * (R[2,1]-R[1,2]) / q[i]
            q[2] = 0.25 * (R[0,1]+R[1,0]) / q[i]
            q[3] = 0.25 * (R[2,0]+R[0,2]) / q[i]

        elif i == 2:
            
            q[0] = 0.25 * (R[0,2]-R[2,0]) / q[i]
            q[1] = 0.25 * (R[0,1]+R[1,0]) / q[i]
            q[3] = 0.25 * (R[2,1]+R[1,2]) / q[i]

        elif i == 3:

            q[0] = 0.25 * (R[1,0]-R[0,1]) / q[i]
            q[1] = 0.25 * (R[0,2]+R[2,0]) / q[i]
            q[2] = 0.25 * (R[2,1]+R[1,2]) / q[i]

        return q

