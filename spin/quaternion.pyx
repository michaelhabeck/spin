import numpy
cimport numpy
cimport cython

from libc.math cimport cos, sin, sqrt, M_PI

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
def matrix(double[::1] q, double[:, ::1] R):

    cdef double r, w, x, y, z

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    r = w**2 + x**2 + y**2 + z**2 + 1e-300

    R[0,0] = ( w**2 + x**2 - y**2 -z**2 ) / r
    R[0,1] = ( 2*x*y - 2*w*z ) / r
    R[0,2] = ( 2*x*z + 2*w*y ) / r

    R[1,0] = ( 2*x*y + 2*w*z ) / r
    R[1,1] = ( w**2 -x**2 +y**2 - z**2 ) / r
    R[1,2] = ( 2*y*z - 2*w*x ) / r

    R[2,0] = ( 2*x*z - 2*w*y ) / r
    R[2,1] = ( 2*y*z + 2*w*x ) / r
    R[2,2] = ( w**2 - x**2 - y**2 + z**2 ) / r

@cython.boundscheck(True)
@cython.wraparound(False)
def params(double[:, ::1] R):

    cdef double a, b, c, d

    a = 0.5 * sqrt(1 + R[0,0] + R[1,1] + R[2,2])
    b = 0.25 * (R[2,1]-R[1,2]) / a
    c = 0.25 * (R[0,2]-R[2,0]) / a
    d = 0.25 * (R[1,0]-R[0,1]) / a

    return a, b, c, d

@cython.boundscheck(True)
@cython.wraparound(False)
def params_(double[:, ::1] R):

    cdef double a, b, c, d

    a = 1 + R[0,0] + R[1,1] + R[2,2]
    b = 1 + R[0,0] - R[1,1] - R[2,2]
    c = 1 - R[0,0] + R[1,1] - R[2,2]
    d = 1 - R[0,0] - R[1,1] + R[2,2]

    if a >= b and a >= c and a >= d:

        a = 0.5 * sqrt(a)
        b = 0.25 * (R[2,1]-R[1,2]) / a
        c = 0.25 * (R[0,2]-R[2,0]) / a
        d = 0.25 * (R[1,0]-R[0,1]) / a

    elif b >= a and b >= c and b >= d: 

        b = 0.5 * sqrt(b)
        a = 0.25 * (R[2,1]-R[1,2]) / b
        c = 0.25 * (R[0,1]+R[1,0]) / b
        d = 0.25 * (R[2,0]+R[0,2]) / b

    elif c >= a and c >= b and c >= d: 

        c = 0.5 * sqrt(c)
        a = 0.25 * (R[0,2]-R[2,0]) / c
        b = 0.25 * (R[0,1]+R[1,0]) / c
        d = 0.25 * (R[2,1]+R[1,2]) / c

    else:

        d = 0.5 * sqrt(d)
        a = 0.25 * (R[1,0]-R[0,1]) / d
        b = 0.25 * (R[0,2]+R[2,0]) / d
        c = 0.25 * (R[2,1]+R[1,2]) / d

    return a, b, c, d

@cython.boundscheck(True)
@cython.wraparound(False)
def jacobian(double[::1] q, double[:, ::1] A, double[:, ::1] B,
             double [:, ::1] C, double [:,::1] D):

    cdef double r, w, x, y, z

    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]

    r = w**2 + x**2 + y**2 + z**2 + 1e-300

    A[0,0] = 4*w*(y**2 + z**2)/r**2
    A[0,1] = 2*(2*w*(w*z - x*y) - z*r)/r**2
    A[0,2] = 2*(-2*w*(w*y + x*z) + y*r)/r**2
    A[1,0] = 2*(-2*w*(w*z + x*y) + z*r)/r**2
    A[1,1] = 4*w*(x**2 + z**2)/r**2
    A[1,2] = 2*(2*w*(w*x - y*z) - x*r)/r**2
    A[2,0] = 2*(2*w*(w*y - x*z) - y*r)/r**2
    A[2,1] = 2*(-2*w*(w*x + y*z) + x*r)/r**2
    A[2,2] = 4*w*(x**2 + y**2)/r**2

    B[0,0] = -2*x*(w**2 + x**2 - y**2 - z**2)/r**2 + 2*x/r
    B[0,1] = -2*x*(-2*w*z + 2*x*y)/r**2 + 2*y/r
    B[0,2] = -2*x*(2*w*y + 2*x*z)/r**2 + 2*z/r
    B[1,0] = -2*x*(2*w*z + 2*x*y)/r**2 + 2*y/r
    B[1,1] = -2*x*(w**2 - x**2 + y**2 - z**2)/r**2 - 2*x/r
    B[1,2] = -2*w/r - 2*x*(-2*w*x + 2*y*z)/r**2
    B[2,0] = -2*x*(-2*w*y + 2*x*z)/r**2 + 2*z/r
    B[2,1] = 2*w/r - 2*x*(2*w*x + 2*y*z)/r**2
    B[2,2] = -2*x*(w**2 - x**2 - y**2 + z**2)/r**2 - 2*x/r

    C[0,0] = -2*y*(w**2 + x**2 - y**2 - z**2)/r**2 - 2*y/r
    C[0,1] = 2*x/r - 2*y*(-2*w*z + 2*x*y)/r**2
    C[0,2] = 2*w/r - 2*y*(2*w*y + 2*x*z)/r**2
    C[1,0] = 2*x/r - 2*y*(2*w*z + 2*x*y)/r**2
    C[1,1] = -2*y*(w**2 - x**2 + y**2 - z**2)/r**2 + 2*y/r
    C[1,2] = -2*y*(-2*w*x + 2*y*z)/r**2 + 2*z/r
    C[2,0] = -2*w/r - 2*y*(-2*w*y + 2*x*z)/r**2
    C[2,1] = -2*y*(2*w*x + 2*y*z)/r**2 + 2*z/r
    C[2,2] = -2*y*(w**2 - x**2 - y**2 + z**2)/r**2 - 2*y/r

    D[0,0] = -2*z*(w**2 + x**2 - y**2 - z**2)/r**2 - 2*z/r
    D[0,1] = -2*w/r - 2*z*(-2*w*z + 2*x*y)/r**2
    D[0,2] = 2*x/r - 2*z*(2*w*y + 2*x*z)/r**2
    D[1,0] = 2*w/r - 2*z*(2*w*z + 2*x*y)/r**2
    D[1,1] = -2*z*(w**2 - x**2 + y**2 - z**2)/r**2 - 2*z/r
    D[1,2] = 2*y/r - 2*z*(-2*w*x + 2*y*z)/r**2
    D[2,0] = 2*x/r - 2*z*(-2*w*y + 2*x*z)/r**2
    D[2,1] = 2*y/r - 2*z*(2*w*x + 2*y*z)/r**2
    D[2,2] = -2*z*(w**2 - x**2 - y**2 + z**2)/r**2 + 2*z/r

