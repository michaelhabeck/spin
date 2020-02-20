"""
Use symbolic math to derive expressions for rotation matrices
based on various parameterizations
"""
import sympy
import sympy.vector as vector

from sympy.matrices import eye
from sympy import cos, sin
from sympy import matrices

def skew_matrix(a):

    return sympy.Matrix([[0, -a[2], a[1]],
                         [a[2], 0, -a[0]],
                         [-a[1], a[0], 0]])

def dyadic_product(a,b):

    return sympy.Matrix([[a[0]*b[0],a[0]*b[1],a[0]*b[2]],
                         [a[1]*b[0],a[1]*b[1],a[1]*b[2]],
                         [a[2]*b[0],a[2]*b[1],a[2]*b[2]]])
                         

theta, phi, alpha = sympy.symbols('theta phi alpha')

xyz = vector.CoordSysCartesian('xyz')

a1, a2, a3 = sympy.symbols('a1 a2 a3')
b1, b2, b3 = sympy.symbols('b1 b2 b3')

a = sympy.Matrix([a1, a2, a3])
b = sympy.Matrix([b1, b2, b3])

from sympy.abc import phi, theta, alpha

axis = sympy.Matrix([cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta)])

R = cos(alpha) * eye(3) + sin(alpha) * skew_matrix(axis) + \
    (1-cos(alpha)) * dyadic_product(axis,axis)

R = sympy.simplify(R)

R2 = sympy.lambdify([theta, phi, alpha], R, "numpy")

import spin

rot = spin.AxisAngle(spin.random_rotation())

print rot.matrix
print R2(*rot.dofs)

RR = matrices.exp(skew_matrix(a))

dR_theta = sympy.Matrix([[sympy.diff(R[i,j], theta) for j in range(3)] for i in range(3)])
#dR_theta = sympy.simplify(dR_theta)

dR_phi = sympy.Matrix([[sympy.diff(R[i,j], phi) for j in range(3)] for i in range(3)])
#dR_phi = sympy.simplify(dR_phi)

dR_alpha = sympy.Matrix([[sympy.diff(R[i,j], alpha) for j in range(3)] for i in range(3)])
#dR_alpha = sympy.simplify(dR_alpha)

## cython code

replacements = [('cos(theta)','ca'),
                ('sin(theta)','sa'),
                ('cos(phi)','cb'),
                ('sin(phi)','sb'),
                ('cos(alpha)','cc'),
                ('sin(alpha)','sc')]

for i in range(3):
    for j in range(3):
        row = str(R[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'R[{0},{1}] = {2}'.format(i,j,row)

for i in range(3):
    for j in range(3):
        row = str(dR_theta[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'A[{0},{1}] = {2}'.format(i,j,row)

for i in range(3):
    for j in range(3):
        row = str(dR_phi[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'B[{0},{1}] = {2}'.format(i,j,row)

for i in range(3):
    for j in range(3):
        row = str(dR_alpha[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'C[{0},{1}] = {2}'.format(i,j,row)

from sympy.abc import w, x, y, z

q = sympy.Matrix([w, x, y, z])
r = w**2 + x**2 + y**2 + z**2
R = sympy.Matrix([[q[0]**2+q[1]**2-q[2]**2-q[3]**2,
                   2*q[1]*q[2]-2*q[0]*q[3],
                   2*q[1]*q[3]+2*q[0]*q[2]],
                  [2*q[1]*q[2]+2*q[0]*q[3],
                   q[0]**2-q[1]**2+q[2]**2-q[3]**2,
                   2*q[2]*q[3]-2*q[0]*q[1]],
                  [2*q[1]*q[3]-2*q[0]*q[2],
                   2*q[2]*q[3]+2*q[0]*q[1],
                   q[0]**2-q[1]**2-q[2]**2+q[3]**2]]) / r
    
dR_w = sympy.Matrix([[sympy.diff(R[i,j], w) for j in range(3)] for i in range(3)])
dR_x = sympy.Matrix([[sympy.diff(R[i,j], x) for j in range(3)] for i in range(3)])
dR_y = sympy.Matrix([[sympy.diff(R[i,j], y) for j in range(3)] for i in range(3)])
dR_z = sympy.Matrix([[sympy.diff(R[i,j], z) for j in range(3)] for i in range(3)])

dR_w = sympy.simplify(dR_w)

replacements = [('w','q[0]'),
                ('x','q[1]'),
                ('y','q[2]'),
                ('z','q[3]')]

replacements = [('w','w'),
                ('x','x'),
                ('y','y'),
                ('z','z'),
                ('(w**2 + x**2 + y**2 + z**2)','r')]

for i in range(3):
    for j in range(3):
        row = str(dR_w[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'A[{0},{1}] = {2}'.format(i,j,row)

print
for i in range(3):
    for j in range(3):
        row = str(dR_x[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'B[{0},{1}] = {2}'.format(i,j,row)

print 
for i in range(3):
    for j in range(3):
        row = str(dR_y[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'C[{0},{1}] = {2}'.format(i,j,row)

print 
for i in range(3):
    for j in range(3):
        row = str(dR_z[i,j])
        for a, b in replacements: row=row.replace(a,b)
        print 'D[{0},{1}] = {2}'.format(i,j,row)

