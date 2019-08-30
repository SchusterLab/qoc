"""
differential_operators.py - This module defines differential operators
to specify autograd computations.
"""

from autograd.core import make_vjp as _make_vjp
from autograd.extend import (vspace,)
from autograd.wrap_util import (unary_to_nary,)
import numpy as np
from pycuda.gpuarray import GPUArray

# This differential operator follows autograd's jacobian implementation.
# https://github.com/HIPS/autograd/blob/master/autograd/differential_operators.py
@unary_to_nary
def ans_jacobian(function, argnum):
    """
    Get the value and the jacobian of a function.
    This differential operator supports numpy and pycuda arrays.

    Args:
    function :: any -> any - the function to differentiate
    argnum :: int - the argument number to differentiate with respect to

    Returns:
    ans_jacobian any -> tuple(any :: any, jacobian :: ndarray) - a function
        that returns the value of `function` and the jacobian
        of `function` evaluated at a given argument of `function`
    """
    vjp, ans = _make_vjp(function, argnum)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(argnum).shape
    grads = list(map(vjp, ans_vspace.standard_basis()))
    if isinstance(grads[0], np.ndarray):
        jacobian = np.reshape(np.stack(grads), jacobian_shape)
    elif isinstance(grads[0], GPUArray):
        jacobian =  stack_gpu(grads).reshape(jacobian_shape)
    
    return ans, jacobian
