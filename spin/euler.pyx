import numpy
cimport numpy
cimport cython

from libc.math cimport cos, sin, atan2
from libc.math cimport M_PI

cdef:
    double TWOPI = 2 * M_PI

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
def matrix(double[::1] dofs, double[:, ::1] R):

    cdef double ca, cb, cc, sa, sb, sc

    ca = cos(dofs[0])
    cb = cos(dofs[1])
    cc = cos(dofs[2])
    sa = sin(dofs[0])
    sb = sin(dofs[1])
    sc = sin(dofs[2])

    R[0,0] = cc * cb * ca - sc * sa
    R[0,1] = cc * cb * sa + sc * ca
    R[0,2] = -cc * sb

    R[1,0] = -sc * cb * ca - cc * sa
    R[1,1] = -sc * cb * sa + cc * ca
    R[1,2] = sc * sb

    R[2,0] = sb * ca
    R[2,1] = sb * sa
    R[2,2] = cb

@cython.boundscheck(True)
@cython.wraparound(False)
def params(double[:, ::1] R):

    cdef double a, b, c

    a = atan2(R[2,1], R[2,0]) % TWOPI
    b = atan2((R[2,0] + R[2,1]) / (cos(a) + sin(a)), R[2,2]) % TWOPI
    c = atan2(R[1,2], -R[0,2]) % TWOPI

    return a, b, c

@cython.boundscheck(True)
@cython.wraparound(False)
def jacobian(double[::1] dofs, double[:, ::1] A, double[:, ::1] B, double [:, ::1] C):

    cdef double ca, cb, cc, sa, sb, sc

    ca = cos(dofs[0])
    cb = cos(dofs[1])
    cc = cos(dofs[2])
    sa = sin(dofs[0])
    sb = sin(dofs[1])
    sc = sin(dofs[2])

    ## alpha

    A[0,0] = -cc * cb * sa - sc * ca
    A[0,1] =  cc * cb * ca - sc *  sa
    A[0,2] = 0

    A[1,0] = sc * cb * sa - cc * ca
    A[1,1] = -sc * cb * ca - cc * sa
    A[1,2] = 0

    A[2,0] = -sb * sa
    A[2,1] =  sb * ca
    A[2,2] = 0

    ## beta

    B[0,0] = -cc * sb * ca
    B[0,1] = -cc * sb * sa
    B[0,2] = -cc * cb

    B[1,0] = sc * sb * ca
    B[1,1] = sc * sb * sa
    B[1,2] = sc * cb

    B[2,0] = cb * ca
    B[2,1] = cb * sa
    B[2,2] =-sb

    ## gamma

    C[0,0] = -sc * cb * ca - cc * sa
    C[0,1] = -sc * cb * sa + cc * ca
    C[0,2] =  sc * sb

    C[1,0] = -cc * cb * ca + sc * sa
    C[1,1] = -cc * cb * sa - sc * ca
    C[1,2] =  cc * sb

    C[2,0] = 0
    C[2,1] = 0
    C[2,2] = 0

