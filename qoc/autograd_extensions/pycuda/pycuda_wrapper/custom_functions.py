"""
custom_functions.py - This module provides definitions of custom functions
that act on pycuda arrays.
"""

import pycuda.gpuarray

_abs_gpu = lambda gpuarray: gpuarray.__abs__()


_power_gpu = lambda gpuarray, exponent: gpuarray.__pow__(exponent)


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


def _swapaxes_gpu(gpuarray, axis1, axis2):
    """
    This function is equivalent to np.swapaxes.

    Arguments:
    gpuarray :: pycuda.gpuarray.GPUArray - the gpuarray whose
        last two axes should be exchanged
    axis1 :: int - axis to be placed where axis2 is
    axis2 :: int - axis to be placed where axis1 is

    Returns:
    gpuarray_swapped :: pycuda.gpuarray.GPUArray - the array
        given where axis1 and axis2 are swapped
    """
    # Genereate a list of the axis indices of the array
    # where the last two indices are swapped.
    new_axes = range(gpuarray.ndim)
    new_axes[axis1] = new_axes[axis2]
    new_axes[axis2] = new_axes[axis1]

    return gpuarray.transpose(axes=new_axes)
