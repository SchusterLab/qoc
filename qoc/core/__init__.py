"""
core - a directory for the primary functionality exposed by qoc
"""

from qoc.core.gsd import grape_schroedinger_discrete
from qoc.core.maths import (interpolate_linear, magnus_m2, magnus_m4, magnus_m6,
                            magnus_m2_linear, magnus_m4_linear, magnus_m6_linear,
                            magnus_m2_linear_param_indices,
                            magnus_m4_linear_param_indices,
                            magnus_m6_linear_param_indices,)

__all__ = [
    "grape_schroedinger_discrete", "interpolate_linear"
    "magnus_m2", "magnus_m4", "magnus_m6", "magnus_m2_linear",
    "magnus_m4_linear", "magnus_m6_linear",
    "magnus_m2_linear_param_indices",
    "magnus_m4_linear_param_indices", "magnus_m6_linear_param_indices",
]
