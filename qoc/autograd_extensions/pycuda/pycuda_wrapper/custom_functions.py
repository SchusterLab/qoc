"""
custom_functions.py - This module provides definitions of custom functions
that act on pycuda arrays.
"""

import skcuda.misc as cumisc
import pycuda.gpuarray

_abs_gpu = lambda a_gpu: a_gpu.__abs__()


def _expand_dims_gpu(a_gpu, axis):
    """
    This function is similar to np.expand_dims.
    
    Arguments: 
    a_gpu :: pycuda.gpuarray.GPUArray - the gpuarray
        to expand
    axis :: int - the axis index to be expanded

    Returns:
    a_gpu_expanded :: pycuda.gpuarray.GPUArray - `a_gpu`
        expanded along `axis`
    """
    shape = list(a_gpu.shape)
    shape.insert(axis, 1)
    new_shape = tuple(shape)

    return _reshape_gpu(a_gpu, new_shape)


_imag_gpu = lambda a_gpu: a_gpu.imag


_iscomplexobj_gpu = lambda a_gpu: cumisc.iscomplextype(a_gpu.dtype)


_power_gpu = lambda a_gpu, exponent: a_gpu.__pow__(exponent)


_real_gpu = lambda a_gpu: a_gpu.real


_reshape_gpu = lambda a_gpu, shape, order="C": a_gpu.reshape(shape, order=order)


def _stack_gpu(a_gpu_list):
    """
    This function is equivalent to np.stack(*args, axis=0) for
    gpu arrays.
    
    Arguments:
    a_gpus :: iterable(pycuda.gpuarray.GPUArray) - the list of
        gpu arrays to stack

    Returns:
    stack :: pycuda.gpuarray.GPUArray - an array where each of the arrays
        in a_gpus is stacked along axis 0
    """
    array_shape = a_gpu_list[0].shape
    array_count = len(a_gpu_list)
    stack_shape = (array_count, *array_shape)
    stack_dtype = a_gpu_list[0].dtype
    stack = pycuda.gpuarray.empty(stack_shape, stack_dtype)
    for i, a_gpu in enumerate(a_gpu_list):
        stack[i] = a_gpu
    
    return stack


_square_gpu = lambda a_gpu: a_gpu * a_gpu


def _squeeze_gpu(a_gpu, axis=None):
    """
    This function is similar to np.squeeze.
    
    Arguments:
    a_gpu :: pycuda.gpuarray.GPUArray - the gpuarray
        to squeeze
    axis :: int - the axis to squeeze

    Returns:
    a_gpu_squeezed :: pycuda.gpuarray.GPUArray - the
        gpuarray squeezed along the given axis
    """
    if ((not (axis is None))
      and a_gpu.shape[axis] != 1):
        raise ValueError("cannot select an axis to squeeze out which "
                         "has size not equal to one")
    elif axis is None:
        return a_gpu.squeeze()
    else:
        new_shape = tuple([dim for i, dim in enumerate(a_gpu.shape) if i != axis])
        new_strides = tuple([a_gpu.strides[i] for i, dim in enumerate(a_gpu.shape) if i != axis])

    return pycuda.gpuarray.GPUArray(shape=new_shape,
                                    dtype=a_gpu.dtype,
                                    allocator=a_gpu.allocator,
                                    strides=new_strides,
                                    base=a_gpu,
                                    gpudata=int(a_gpu.gpudata))
    


def _swapaxes_gpu(a_gpu, axis1, axis2):
    """
    This function is equivalent to np.swapaxes.

    Arguments:
    a_gpu :: pycuda.gpuarray.GPUArray - the gpuarray whose
        last two axes should be exchanged
    axis1 :: int - axis to be placed where axis2 is
    axis2 :: int - axis to be placed where axis1 is

    Returns:
    a_gpu_swapped :: pycuda.gpuarray.GPUArray - the array
        given where axis1 and axis2 are swapped
    """
    # Genereate a list of the axis indices of the array
    # where the last two indices are swapped.
    new_axes = range(a_gpu.ndim)
    new_axes[axis1] = new_axes[axis2]
    new_axes[axis2] = new_axes[axis1]

    return a_gpu.transpose(axes=new_axes)


_where_gpu = lambda criterion, *args, **kwargs: pycuda.gpuarray.if_positive(_abs_gpu(criterion),
                                                                            *args, **kwargs)
