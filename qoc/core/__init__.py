"""
core - a directory for the primary functionality exposed by qoc
"""

from grape import (grape_schroedinger_discrete)
from maths import (interpolate_trapezoid, magnus_m2, magnus_m4, magnus_m6)

__all__ = ["grape_schroedinger_discrete", "interpolate_trapezoid",
           "magnus_m2", "magnus_m4", "magnus_m6"]
