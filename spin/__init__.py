"""
spin library
------------

Three-dimensional rotation matrices and various parameterizations including
* EulerAngles
* AxisAngle
* ExponentialMap
* Quaternion
"""
import numpy as np

from .rigid import RigidTransformation
from .trafo import Translation, compose
from .fitting import qfit
from .fitting import LeastSquares, NearestRotation, NearestUnitQuaternion, NearestQuaternion
from .rotation import Rotation, Parameterization, EulerAngles
from .rotation import AxisAngle, ExponentialMap, Quaternion
from .rotation import Azimuth, Polar, RotationAngle
from .rotation import random_rotation, distance

