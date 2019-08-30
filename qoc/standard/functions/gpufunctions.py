"""
gpufunctions.py - This module provides a common namespace for all gpu functions.

NOTE:
skcuda.misc functions are preferred over the pycuda.gpuarray.GPUArray object
methods because skcuda supports array broadcasting like numpy. Furthermore,
we want the gpu functions to have the same behavior as their cpu counterparts.
"""

from autograd.extend import primitive
import skcuda.linalg as culinalg
import skcuda.misc as cumisc

abs_gpu = lambda x: x.abs()

add_gpu = cumisc.add

conj_gpu = lambda x: x.conj()

divide_gpu = cumisc.divide

matmul_gpu = cumisc.dot

multiply_gpu = cumisc.multiply

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

subtract_gpu = cumisc.subtract

sum_gpu = cumisc.sum

trace_gpu = culinalg.trace

transpose_gpu = culinalg.transpose
