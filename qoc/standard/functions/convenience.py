"""
convenience.py - definitions of common computations
All functions in this module that are exported, 
i.e. those that don't begin with '_', are autograd compatible.
"""

from functools import reduce

import autograd.numpy as anp

from qoc.models.operationpolicy import OperationPolicy
from qoc.autograd_extensions.pycuda.pycudawrapper import (matmul_gpu, sum_gpu,)

### COMPUTATIONS ###

def commutator(a, b):
    """
    Compute the commutator of two matrices.

    Arguments:
    a :: numpy.ndarray - the left matrix
    b :: numpy.ndarray - the right matrix

    Returns:
    _commutator :: numpy.ndarray - the commutator of a and b
    """
    if operation_policy == OperationPolicy.CPU:
        commutator_ = anp.matmul(a, b) - anp.matmul(b, a)
    elif operation_policy == OperationPolicy.GPU:
        commutator_ = matmul_gpu(a, b) - matmul_gpu(b, a)
    else:
        _not_implemented(operation_policy)

    return commutator_

def conjugate(x, operation_policy=OperationPolicy.CPU):
    """
    Compute the conjugate of a value.
    
    Args:
    x :: ndarray - the value to compute the conjugate of
    
    Returns:
    conj :: ndarray - the conjugate of x
    """
    if operation_policy == OperationPolicy.CPU:
        conj = anp.conjugate(x)
    elif operation_policy == OperationPolicy.GPU:
        conj = conj_gpu(x)
    else:
        _not_implemented(operation_policy)

    return conj


def conjugate_transpose(a, operation_policy=OperationPolicy.CPU):
    """
    Compute the conjugate transpose of a matrix.

    Args:
    a :: ndarray - the array to compute the conjugate transpose of
    operation_policy

    Returns:
    _conjugate_tranpose :: ndarray - the conjugate transpose of x
    """
    return conjugate(transpose(a, operation_policy=operation_policy),
                     operation_policy=operation_policy)


def krons(*matrices):
    """
    Compute the kronecker product of a list of matrices.
    Args:
    matrices :: numpy.ndarray - the list of matrices to
        compute the kronecker product of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    """
    if operation_policy == OperationPolicy.CPU:
        krons_ = reduce(anp.kron, matrices)
    else:
        _not_implemented(operation_policy)

    return krons_


def matmuls(*matrices, operation_policy=OperationPolicy.CPU):
    """
    Compute the kronecker product of a list of matrices.
    Args:
    matrices :: numpy.ndarray - the list of matrices to
        compute the kronecker product of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    """
    if operation_policy == OperationPolicy.CPU:
        matmul = anp.matmul
    elif  operation_policy == OperationPolicy.GPU:
        matmul = matmul_gpu
    else:
        _not_implemented(operation_policy)

    return reduce(matmul, matrices)


def rms_norm(array, operation_policy=OperationPolicy.CPU):
    """
    Compute the rms norm of the array.

    Arguments:
    array :: ndarray (N) - The array to compute the norm of.

    Returns:
    norm :: float - The rms norm of the array.
    """
    if operation_policy == OperationPolicy.CPU:
        square_norm = anp.sum(anp.real(array * anp.conjugate(array)))
        size = anp.prod(anp.shape(array))
        rms_norm_ = anp.sqrt(square_norm / size)
    else:
        _not_implemented(operation_policy)
        
    return rms_norm_


def mult_rows(matrix, vector, operation_policy=OperationPolicy.CPU):
    """
    Multiply each row vector in `matrix` by the corresponding element
    in `vector`.
    Args:
    matrix :: numpy.ndarray - an N x N matrix
    vector :: numpy.ndarray - an N vector
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _matrix :: numpy.ndarray - the requested matrix
    """
    if operation_policy == OperationPolicy.CPU:
        matrix_ = transpose(transpose(matrix, operation_policy)
                            * vector, operation_policy)
    else:
        _not_implemented(operation_policy)

    return matrix_


def square(a, operation_policy=OperationPolicy.CPU):
    """
    Square an array.

    Args:
    a :: ndarray - the array to square
    
    Returns
    a_squared :: ndarray - the square of `a`
    """
    if operation_policy == OperationPolicy.CPU:
        a_squared =  anp.square(a)
    elif operation_policy == OperationPolicy.GPU:
        a_squared = square_gpu(a)
    else:
        return 


def sum_axis(*args, **kwargs):
    """
    This function is similar to np.sum and skcuda.misc.sum.

    Arguments:
    operation_policy

    Returns:
    sum :: ndarray - the specified sum
    """
    if "operation_policy" in kwargs:
        operation_policy = kwargs.pop("operation_policy")
    else:
        operation_policy = OperationPolicy.CPU
    
    if operation_policy == OperationPolicy.CPU:
        sum_ = anp.sum(*args, **kwargs)
    elif operation_policy == OperationPolicy.GPU:
        sum_ = sum_gpu(*args, **kwargs)
    else:
        _not_implemented(operation_policy)

    return sum_


def transpose(a, operation_policy=OperationPolicy.CPU):
    """
    Obtain the transpose of the array.

    Args:
    a :: ndarray - the array to compute the transpose of
    operation_policy

    Returns:
    a_transpose :: ndarray - the transpose of `a`
    """
    if operation_policy == OperationPolicy.CPU:
        swapaxes = anp.swapaxes
    elif operation_policy == OperationPolicy.GPU:
        swapaxes = swapaxes_gpu
    else:
        _not_implemented(operation_policy)

    return swapaxes(a, -1, -2)


### ISOMORPHISMS ###

# A row vector is np.array([[0, 1, 2]])
# A column vector is np.array([[0], [1], [2]])
column_vector_list_to_matrix = (lambda column_vector_list:
                                anp.hstack(column_vector_list))


matrix_to_column_vector_list = (lambda matrix:
                                anp.stack([anp.vstack(matrix[:, i])
                                           for i in range(matrix.shape[1])]))

# Take a flat array of scalars from C to R2.
complex_to_real_imag_flat = lambda x: np.hstack((np.real(x), np.imag(x)))


def real_imag_to_complex_flat(x):
    """
    Take a flat array of scalars from R2 to C.
    Args:
    x :: numpy.ndarray - the flat array of real scalars to map
    Returns:
    x_c :: numpy.ndarray - the complex scalars mapped from x
    """
    real, imag = np.split(x, 2)
    return real + 1j * imag


### HELPER FUNCTIONS ###

def _not_implemented(operation_policy):
    """
    Raise a NotImplementedError for the given operation policy.
    """
    raise NotImplementedError("The requested operation is not implemented "
                              "for the {} operation policy."
                              "".format(operation_policy))



### MODULE TESTS ###

_BIG = 100

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: nothing
    """

    # Test row and column manipulations.
    matrix = np.random.rand(matrix_size, matrix_size)
    vector = np.random.rand(matrix_size)

    col_mult_matrix = np.zeros_like(matrix)
    for col_index in range(matrix_size):
        col_mult_matrix[:, col_index] = matrix[:, col_index] * vector[col_index]

    row_mult_matrix = np.zeros_like(matrix)
    for row_index in range(matrix_size):
        row_mult_matrix[row_index, :] = matrix[row_index, :] * vector[row_index]

    assert(np.allclose(col_mult_matrix, mult_cols(matrix, vector)))
    assert(np.allclose(row_mult_matrix, mult_rows(matrix, vector)))
    
    # Test complex number mapping.
    rand_complex = np.random.rand(_BIG) + 1j * np.random.rand(_BIG)
    rand_complex_mapped = real_imag_to_complex_flat(complex_to_real_imag_flat(rand_complex))
    assert(np.allclose(rand_complex, rand_complex_mapped))


if __name__ == "__main__":
    _tests()
=======
>>>>>>> dev
