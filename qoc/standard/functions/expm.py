"""
expm.py - a module for all things e^M
"""

from autograd.extend import (defvjp as autograd_defvjp,
                             primitive as autograd_primitive)
import numpy as np
import scipy.linalg as la

@autograd_primitive
def expm(matrix):
    """
    Compute the matrix exponential of a matrix.
    Args:
    matrix :: numpy.ndarray - the matrix to exponentiate
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    exp_matrix :: numpy.ndarray - the exponentiated matrix
    """
    exp_matrix = la.expm(matrix)

    return exp_matrix


def _expm_vjp(exp_matrix, matrix):
    """
    Construct the left-multiplying vector jacobian product function
    for the matrix exponential.

    Intuition:
    `dfinal_dexpm` is the jacobian of `final` with respect to each element `expmij`
    of `exp_matrix`. `final` is the output of the first function in the
    backward differentiation series. It is also the output of the last
    function evaluated in the chain of functions that is being differentiated,
    i.e. the final cost function. The goal of `vjp_function` is to take
    `dfinal_dexpm` and yield `dfinal_dmatrix` which is the jacobian of
    `final` with respect to each element `mij` of `matrix`.
    To compute the frechet derivative of the matrix exponential with respect
    to each element `mij`, we use the approximation that
    dexpm_dmij = np.matmul(Eij, exp_matrix). Since Eij has a specific
    structure we don't need to do the full matrix multiplication and instead
    use some indexing tricks.

    Args:
    exp_matrix :: numpy.ndarray - the matrix exponential of matrix
    matrix :: numpy.ndarray - the matrix that was exponentiated
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method

    Returns:
    vjp_function :: numpy.ndarray -> numpy.ndarray - the function that takes
        the jacobian of the final function with respect to `exp_matrix`
        to the jacobian of the final function with respect to `matrix`
    """
    matrix_size = matrix.shape[0]
        
    def _expm_vjp_(dfinal_dexpm):
        dfinal_dmatrix = np.zeros((matrix_size, matrix_size), dtype=np.complex128)

        # Compute a first order approximation of the frechet derivative of the matrix
        # exponential in every unit direction Eij.
        for i in range(matrix_size):
            for j in range(matrix_size):
                dexpm_dmij_rowi = exp_matrix[j,:]
                dfinal_dmatrix[i, j] = np.sum(np.multiply(dfinal_dexpm[i, :], dexpm_dmij_rowi))
            #ENDFOR
        #ENDFOR

        return dfinal_dmatrix
    #ENDDEF

    return _expm_vjp_


autograd_defvjp(expm, _expm_vjp)


### MODULE TESTS ###

_BIG = 30

def _get_skew_hermitian_matrix(matrix_size):
    """
    Args:
    matrix_size :: int - square matrix size

    Returns:
    skew_hermitian_matrix :: numpy.ndarray - a skew-hermitian matrix
        of `matrix_size`
    """
    matrix = (np.random.rand(matrix_size, matrix_size)
              + 1j * np.random.rand(matrix_size, matrix_size))
    
    return np.divide(matrix - conjugate_transpose(matrix), 2)

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: none
    """
    from autograd import jacobian
    # Test that the end-to-end gradient of the matrix exponential is working.
    m = np.array([[1., 0.],
                  [0., 1.]])
    m_len = m.shape[0]
    exp_m = np.exp(m)
    dexpm_dm_expected = np.zeros((m_len, m_len, m_len, m_len), dtype=m.dtype)
    dexpm_dm_expected[0, 0, 0, 0] = exp_m[0, 0]
    dexpm_dm_expected[0, 1, 0, 1] = exp_m[0, 0]
    dexpm_dm_expected[1, 0, 1, 0] = exp_m[1, 1]
    dexpm_dm_expected[1, 1, 1, 1] = exp_m[1, 1]
    
    dexpm_dm = jacobian(expm, 0)(m)

    assert(np.allclose(dexpm_dm, dexpm_dm_expected))


if __name__ == "__main__":
    _tests()
