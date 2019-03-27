import numpy
cimport numpy
cimport cython

from libc.math cimport cos, sin, acos, atan2, M_PI

DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t

DTYPE_INT = numpy.int
ctypedef numpy.int_t DTYPE_INT_t

DTYPE_DOUBLE = numpy.double
ctypedef numpy.double_t DTYPE_DOUBLE_t

DTYPE_LONG = numpy.long
ctypedef numpy.long_t DTYPE_LONG_t

@cython.boundscheck(True)
@cython.wraparound(False)
def axis(double theta, double phi, double[::1] axis):

    cdef double c, s, C, S

    c = cos(phi)
    s = sin(phi)
    C = cos(theta)
    S = sin(theta)

    axis[0] = c * S
    axis[1] = s * S
    axis[2] = C

@cython.boundscheck(True)
@cython.wraparound(False)
def matrix(double[::1] dofs, double[:, ::1] R):

    cdef double a[3]
    cdef double c, s, C, S

    c = cos(dofs[1])
    s = sin(dofs[1])
    C = cos(dofs[0])
    S = sin(dofs[0])

    a[0] = c * S
    a[1] = s * S
    a[2] = C

    c = cos(dofs[2])
    s = sin(dofs[2])

    R[0,0] = c + (1-c) * a[0] * a[0]
    R[1,1] = c + (1-c) * a[1] * a[1]
    R[2,2] = c + (1-c) * a[2] * a[2]

    R[0,1] = -s * a[2] + (1-c) * a[0] * a[1]
    R[0,2] =  s * a[1] + (1-c) * a[0] * a[2]

    R[1,0] =  s * a[2] + (1-c) * a[1] * a[0]
    R[1,2] = -s * a[0] + (1-c) * a[1] * a[2]

    R[2,0] = -s * a[1] + (1-c) * a[2] * a[0]
    R[2,1] =  s * a[0] + (1-c) * a[2] * a[1]
    
@cython.boundscheck(True)
@cython.wraparound(False)
def params(double[:, ::1] R):

    cdef double angle, phi, theta

    angle = acos(0.5 * (R[0,0] + R[1,1] + R[2,2] - 1))

    if angle != 0.:
        theta = acos(0.5 * (R[1,0] - R[0,1]) / sin(angle))
        phi   = atan2(R[0,2]-R[2,0], R[2,1]-R[1,2])

    else:
        theta = 0
        phi   = 0

    return theta, phi, angle

@cython.boundscheck(True)
@cython.wraparound(False)
def jacobian(double[::1] dofs, double[:, ::1] A, double[:, ::1] B, double [:, ::1] C):

    cdef double ca, cb, cc, sa, sb, sc, pi

    pi = M_PI

    ca = cos(dofs[0])
    cb = cos(dofs[1])
    cc = cos(dofs[2])

    sa = sin(dofs[0])
    sb = sin(dofs[1])
    sc = sin(dofs[2])

    A[0,0] = 2*(-cc + 1)*sa*cb**2*ca
    A[0,1] = -2*(cc - 1)*sb*sa*cb*ca + sc*sa
    A[0,2] = (-(cc - 1)*cb*ca + sc*sb)*ca + (cc - 1)*sa**2*cb
    A[1,0] = -2*(cc - 1)*sb*sa*cb*ca - sc*sa
    A[1,1] = 2*(-cc + 1)*sb**2*sa*ca
    A[1,2] = -((cc - 1)*sb*ca + sc*cb)*ca + (cc - 1)*sb*sa**2
    A[2,0] = -((cc - 1)*cb*ca + sc*sb)*ca + (cc - 1)*sa**2*cb
    A[2,1] = (-(cc - 1)*sb*ca + sc*cb)*ca + (cc - 1)*sb*sa**2
    A[2,2] = -2*(-cc + 1)*sa*ca

    B[0,0] = 2*(cc - 1)*sb*sa**2*cb
    B[0,1] = 2*(-cc + 1)*sa**2*sin(dofs[1] + pi/4)*cos(dofs[1] + pi/4)
    B[0,2] = ((cc - 1)*sb*ca + sc*cb)*sa
    B[1,0] = 2*(-cc + 1)*sa**2*sin(dofs[1] + pi/4)*cos(dofs[1] + pi/4)
    B[1,1] = 2*(-cc + 1)*sb*sa**2*cb
    B[1,2] = -((cc - 1)*cb*ca - sc*sb)*sa
    B[2,0] = ((cc - 1)*sb*ca - sc*cb)*sa
    B[2,1] = -((cc - 1)*cb*ca + sc*sb)*sa
    B[2,2] = 0

    C[0,0] = (sa**2*cb**2 - 1)*sc
    C[0,1] = sc*sb*sa**2*cb - cc*ca
    C[0,2] = (sc*cb*ca + sb*cc)*sa
    C[1,0] = sc*sb*sa**2*cb + cc*ca
    C[1,1] = (sb**2*sa**2 - 1)*sc
    C[1,2] = (sc*sb*ca - cc*cb)*sa
    C[2,0] = (sc*cb*ca - sb*cc)*sa
    C[2,1] = (sc*sb*ca + cc*cb)*sa
    C[2,2] = -sc*sa**2

