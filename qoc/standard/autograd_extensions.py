"""
autograd_extensions.py - a module for extending autograd's core
functionality
"""

from autograd.wrap_util import unary_to_nary
from autograd.core import make_vjp as _make_vjp
from autograd.extend import vspace

import numpy as np

### DIFFERENTIAL OPERATORS ###

# This mostly follows autograd's jacobian implementation:
# https://github.com/HIPS/autograd/blob/master/autograd/differential_operators.py
# but returns the value of the function in addition to the jacobian.
@unary_to_nary
def ans_jacobian(function, argnum):
    """
    Get the value and the jacobian of a function.
    Args:
    function :: any -> any - the function to differentiate
    argnum :: int - the argument number to differentiate with respect to
    Returns:
    ans_jacobian any -> tuple(any :: any, jacobian :: any) - a function
        that returns the value of "function" and the jacobian
        of "function" evaluated at a given argument of "function"
    """
    vjp, ans = _make_vjp(function, argnum)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(argnum).shape
    grads = list(map(vjp, ans_vspace.standard_basis()))
    
    return ans, np.reshape(np.stack(grads), jacobian_shape)
