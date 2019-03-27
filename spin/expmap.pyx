import numpy
cimport numpy
cimport cython

from libc.math cimport cos, sin, acos, atan2, sqrt

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
def matrix(double[::1] a, double[:, ::1] R):

    cdef double theta, c, s

    R[0,0] = 1
    R[1,1] = 1
    R[2,2] = 1
    
    theta = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

    if theta > 1e-300:

        c = (1 - cos(theta)) / theta / theta
        s = sin(theta) / theta

        R[0,0] += c * (a[0] * a[0] - theta * theta)
        R[1,1] += c * (a[1] * a[1] - theta * theta)
        R[2,2] += c * (a[2] * a[2] - theta * theta)

        R[0,1] = -s * a[2] + c * a[0] * a[1] 
        R[0,2] =  s * a[1] + c * a[0] * a[2] 

        R[1,0] =  s * a[2] + c * a[1] * a[0] 
        R[1,2] = -s * a[0] + c * a[1] * a[2] 

        R[2,0] = -s * a[1] + c * a[2] * a[0] 
        R[2,1] =  s * a[0] + c * a[2] * a[1] 

    else:

        R[0,1] = 0
        R[0,2] = 0

        R[1,0] = 0
        R[1,2] = 0

        R[2,0] = 0
        R[2,1] = 0
    
@cython.boundscheck(True)
@cython.wraparound(False)
def params(double[:, ::1] R):

    cdef double x, y, z, f, angle
    
    angle = acos(0.5 * (R[0,0] + R[1,1] + R[2,2] - 1))

    if angle != 0.:
        f = 0.5 * angle / sin(angle)
        x = f * (R[2,1] - R[1,2])
        y = f * (R[0,2] - R[2,0])
        z = f * (R[1,0] - R[0,1])

    else:
        x = 0
        y = 0
        z = 0

    return x, y, z

