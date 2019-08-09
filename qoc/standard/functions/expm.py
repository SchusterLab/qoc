"""
expm.py - a module for all things e^M
"""

from autograd import jacobian
from autograd.extend import (defvjp as autograd_defvjp,
                             primitive as autograd_primitive)
from jax import (custom_transforms as jax_primitive,
                 defvjp as jax_defvjp)

import numpy as np
import scipy.linalg as la

from qoc.models.operationpolicy import OperationPolicy
from qoc.standard import get_eij
from qoc.standard.autograd_extensions import ans_jacobian

#@jax_primitive
@autograd_primitive
def expm(matrix, operation_policy=OperationPolicy.CPU):
    """
    Compute the matrix exponential of a matrix.
    Args:
    matrix :: numpy.ndarray - the matrix to exponentiate
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _expm :: numpy.ndarray - the exponentiated matrix
    """
    if operation_policy == OperationPolicy.CPU:
        _expm = la.expm(matrix)
    else:
        pass

    return _expm


def _expm_vjp(_expm, matrix, operation_policy=OperationPolicy.CPU):
    """
    Construct the left-multiplying vector jacobian product function
    for the matrix exponential.

    IMPLEMENTATION NOTE: The dexpm_dmij line could be moved out
    of the inner i, j loop and instead stored in the jacobian
    dexpm_dm. However, dexpm_dm is (n x n) x (n x n) and would
    take up considerable memory. Because we do not expect
    to compute the jacobian dexpm_dm and instead expect
    to compute the intermediate vector-jacobian-product where
    the final function has a scalar output (e.g. GRAPE) we
    choose to leave the dexpm_dmij line in the inner i, j loop.

    Args:
    _expm :: numpy.ndarray - the matrix exponential of matrix
    matrix :: numpy.ndarray - the matrix that was exponentiated
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method

    Returns:
    vjp_function :: numpy.ndarray -> numpy.ndarray - the function that takes
        the gradient of the final function, with respect to _expm,
        to the gradient of the final function with respect to "matrix"
    """
    if operation_policy == OperationPolicy.CPU:
        def _expm_vjp_cpu(dfinal_dexpm):
            # Important to use np.zeros_like here to infer datatype.
            dfinal_dmatrix = np.zeros_like(dfinal_dexpm)
            # matrix_size is equivalent to dfinal_dmatrix_size,
            # _expm_size, or dfinal_dexpm_size.
            matrix_size = len(matrix)
            for i in range(matrix_size):
                for j in range(matrix_size):
                    # dexpm_dmij is the jacobian of expm with respect to the mij element
                    # of the matrix.
                    eij = get_eij(i, j, matrix_size)
                    dexpm_dmij = la.expm_frechet(matrix, eij,
                                                 compute_expm=False)
                    # The contribution of the mij element of the matrix to the final function
                    # is the sum of the contributions of mij to each element expmij
                    # of _expm.
                    dfinal_dmatrix[i, j] = np.sum(np.multiply(dfinal_dexpm, dexpm_dmij))
                #ENDFOR
            #ENDFOR

            return dfinal_dmatrix
        #ENDDEF

        vjp_function = _expm_vjp_cpu
    else:
        pass

    return vjp_function


autograd_defvjp(expm, _expm_vjp)
# jax_defvjp(expm, _expm_vjp)


### MODULE TESTS ###

_BIG = 100

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: nothing
    """

    # Test that the gradient of the matrix exponential is working.
    m = np.array([[1, 0],
                  [0, 1]])
    _expm, dexpm_dm = jacobian(expm, 0)(m)
    print("expm:\n{}"
          "".format(_expm))
    print("dexpm_dm:\n{}"
          "".format(dexpm_dm))


if __name__ == "__main__":
    _tests()
