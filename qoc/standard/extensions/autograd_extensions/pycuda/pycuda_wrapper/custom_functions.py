"""
custom_functions.py - This module provides definitions of custom functions
that act on pycuda arrays.
"""

import pycuda.gpuarray

_abs_gpu = lambda gpuarray: gpuarray.__abs__()


_power_gpu = lambda gpuarray, power: gpuarray.__pow__(power)


def _stack_gpu(gpuarrays):
    """
    This function is equivalent to np.stack(*args, axis=0) for
    gpu arrays.
    
    Arguments:
    gpuarrays :: iterable(pycuda.gpuarray.GPUArray) - the list of
        gpu arrays to stack

    Returns:
    stack :: pycuda.gpuarray.GPUArray - an array where each of the arrays
        in gpuarrays is stacked along axis 0
    """
    array_shape = gpuarray_list[0].shape
    array_count = len(gpuarray_list)
    stack_shape = (array_count, *array_shape)
    stack_dtype = gpuarray_list[0].dtype
    stack = pycuda.gpuarray.empty(stack_shape, stack_dtype)
    for i, gpuarray in enumerate(gpuarray_list):
        stack[i] = gpuarray
    
    return stack
