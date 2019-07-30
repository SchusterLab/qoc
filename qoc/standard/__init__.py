"""
standard - a directory for standard definitions
"""

# exports

from qoc.standard.adam import Adam
from qoc.standard.targetinfidelity import TargetInfidelity
from qoc.standard.forbidstates import ForbidStates

# autograd definitions
from autograd.extend import defvjp, primitive
import autograd.numpy as anp
import scipy.linalg as la

# expm
@primitive
def expm(x):
    return la.expm(x)


def expm_vjp(ans, x):
    return lambda g: anp.matmul(g, la.expm_frechet(x, x, compute_expm=False))


defvjp(expm, expm_vjp)


__all__ = [
    "Adam", "TargetInfidelity", "ForbidStates",
]
