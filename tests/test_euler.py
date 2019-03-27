"""
Various tests of the Euler angle parameterization
"""
import spin
import numpy as np
import csb.numeric as csb
import unittest

from spin import euler as eulerlib

## standard rotations about x-y-z axis

def rotation_about_axis(angle, axis='x'):
    matrix = csb.rotation_matrix(np.eye(3)['xyz'.index(axis)], angle)
    return spin.Rotation(matrix)

def R_x(angle):
    return rotation_about_axis(angle, 'x')

def R_y(angle):
    return rotation_about_axis(angle, 'y')

def R_z(angle):
    return rotation_about_axis(angle, 'z')

def compose_euler(a, b, c):
    return R_z(c)(R_y(b)(R_z(a)))

class TestEuler(unittest.TestCase):

    tol = 1e-10

    def test_matrix(self):
        
        angles = np.random.random(3)
        euler  = spin.EulerAngles(angles)
        euler2 = spin.EulerAngles.from_rotation(euler)
        euler3 = compose_euler(*angles)

        a, b, c = angles
        euler4  = spin.compose(R_z(c),R_y(b),R_z(a))

        R = np.zeros((3,3))

        eulerlib.matrix(angles, R)

        self.assertTrue(spin.distance(euler, euler2) < self.tol)
        self.assertTrue(spin.distance(euler, euler3) < self.tol)
        self.assertTrue(spin.distance(euler, euler4) < self.tol)
        self.assertTrue(spin.distance(euler, R) < self.tol)

    def test_angles(self):

        angles = spin.EulerAngles.random()
        euler  = spin.EulerAngles(angles)

        self.assertTrue(np.linalg.norm(angles - euler.dofs) < self.tol)
        self.assertTrue(np.linalg.norm(angles - eulerlib.params(euler.matrix)) < self.tol)

    def test_compose(self):

        n = 10
        R = [spin.EulerAngles() for _ in range(n)]

        M = np.eye(3)
        for R_ in R:
            M = M.dot(R_.matrix)

        R_tot = spin.compose(*R)

        self.assertTrue(spin.distance(M, R_tot) < self.tol)
        
if __name__ == '__main__':
    unittest.main()
