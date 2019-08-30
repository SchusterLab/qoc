"""
convenience.py - definitions of common computations
All functions in this module that are exported, 
i.e. those that don't begin with '_', are autograd compatible.
"""

from functools import reduce

import autograd.numpy as anp
import numpy as np
import scipy.linalg as la
import skcuda.misc as cumisc

from qoc.models.operationpolicy import OperationPolicy

### COMPUTATIONS ###

def commutator(a, b, operation_policy=OperationPolicy.CPU):
    """
    Compute the commutator of two matrices.
    Args:
    a :: numpy.ndarray - the left matrix
    b :: numpy.ndarray - the right matrix
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _commutator :: numpy.ndarray - the commutator of a and b
    """
    if operation_policy == OperationPolicy.CPU:
        _commutator = anp.matmul(a, b) - anp.matmul(b, a)
    else:
        pass

    return _commutator


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
    else:
        pass

    return conj

def conjugate_transpose(matrix, operation_policy=OperationPolicy.CPU):
    """
    Compute the conjugate transpose of a matrix.
    Args:
    matrix :: numpy.ndarray - the matrix to compute
        the conjugate transpose of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _conjugate_tranpose :: numpy.ndarray the conjugate transpose
        of matrix
    """
    if operation_policy == OperationPolicy.CPU:
        _conjugate_transpose = anp.conjugate(transpose(matrix,
                                                       operation_policy))
    else:
        pass
        
    return _conjugate_transpose


def krons(*matrices, operation_policy=OperationPolicy.CPU):
    """
    Compute the kronecker product of a list of matrices.
    Args:
    matrices :: numpy.ndarray - the list of matrices to
        compute the kronecker product of
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    """
    if operation_policy == OperationPolicy.CPU:
        _krons = reduce(anp.kron, matrices)
    else:
        pass

    return _krons


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
        _matmuls = reduce(anp.matmul, matrices)
    else:
        pass

    return _matmuls


def mult_cols(matrix, vector, operation_policy=OperationPolicy.CPU):
    """
    Multiply each column vector in `matrix` by the corresponding
    element in `vector`.
    Args:
    matrix :: numpy.ndarray - an N x N matrix
    vector :: numpy.ndarray - an N vector
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _matrix :: numpy.ndarray - the requested matrix
    """
    if operation_policy == OperationPolicy.CPU:
        _matrix = matrix * vector
    else:
        pass

    return _matrix


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
        _matrix = transpose(transpose(matrix, operation_policy)
                            * vector, operation_policy)
    else:
        pass

    return _matrix

def stack_gpu(gpuarray_list):
    """
    This function is equivalent to np.stack(*args, axis=0) for
    gpu arrays.
    
    Arguments:
    gpuarray_list :: list(pycuda.gpuarray.GPUArray) - the list of
        gpu arrays to stack

    Returns:
    stack :: pycuda.gpuarray.GPUArray - an array where each of the arrays
        in gpuarray_list is stacked along axis 0
    """
    array_shape = gpuarray_list[0].shape
    array_count = len(gpuarray_list)
    stack_shape = (array_count, *array_shape)
    stack_dtype = gpuarray_list[0].dtype
    stack = pycuda.gpuarray.empty(stack_shape, stack_dtype)
    for i, gpuarray in enumerate(gpuarray_list):
        stack[i] = gpuarray
    
    return stack


def sum_axis(array, *args, **kwargs):
    """
    This function is a wrapper around np.sum and skcuda.misc.sum.

    Arguments:
    array :: ndarray - the array to sum over
    operation_policy

    Returns:
    sum :: ndarray - the specified sum
    """
    if "operation_policy" in kwargs:
        operation_policy = kwargs["operation_policy"]
    else:
        operation_policy = OperationPolicy.CPU
    
    if operation_policy == OperationPolicy.CPU:
        _sum = anp.sum(array, *args, **kwargs)
    elif operation_policy == OperationPolicy.GPU:
        _sum = cumisc.sum(array, *args, **kwargs)
    else:
        raise ValueError("This operation is not implemented for "
                         "the operation policy {}."
                         "".format(operation_policy))
    return _sum


def transpose(matrix, operation_policy=OperationPolicy.CPU):
    """
    Obtain the transpose of the matrix.
    Args:
    matrix :: numpy.ndarray - an N x M matrix
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    matrix_transpose :: numpy.ndarray - an M x N matrix that is the
        transpose of `matrix`
    """
    if operation_policy == OperationPolicy.CPU:
        matrix_transpose = anp.swapaxes(matrix, -1, -2)
    else:
        pass

    return matrix_transpose


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
