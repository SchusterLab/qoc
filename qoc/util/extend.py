"""
extend.py - a utility file to extend external packages and libraries
"""

from autograd.wrap_util import unary_to_nary
from autograd.core import make_vjp as _make_vjp

import numpy as np

# autograd extensions

make_vjp = unary_to_nary(_make_vjp)

@unary_to_nary
def ans_jacobian(f, x):
    """
    Get the value and the jacobian of a function.
    Args:
    f :: any -> any - the function to differentiate
    x :: int - the argument number to differentiate with respect to
    Returns:
    ans_jacobian any -> ans :: any, jacobian :: any - a function
        that returns the value of "f" and the jacobian
        of "f"
    """
    vjp, ans = _make_vjp(fun, x)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return ans, np.reshape(np.stack(grads), jacobian_shape)
