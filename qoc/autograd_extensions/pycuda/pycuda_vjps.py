"""
pycuda_vjps.py - This module defines vector jacobian products for the functions
under the common namespace.

NOTE:
These definitions are analagous to their numpy counterparts:
https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py
"""

import numpy as onp

from qoc.standard.extensions.autograd_extensions.pycuda.pycuda_wrapper import (
    abs_gpu, add_gpu, conj_gpu, divide_gpu, expand_dims_gpu, iscomplexobj_gpu, log_gpu, matmul_gpu, multiply_gpu, 
    ones_like_gpu, reshape_gpu, squeeze_gpu ,stack_gpu, subtract_gpu, sum_gpu, trace_gpu, where_gpu,
    zeros_gpu,
)

### VJPS ###

DIFFERENTIABLE_GPU_FUNCTIONS = (
    abs_gpu, add_gpu, conj_gpu, divide_gpu, matmul_gpu, multiply_gpu,
    square_gpu, subtract_gpu, sum_gpu, swapaxes_gpu,
)

defvjp(abs_gpu,
       lambda ans, x: lambda g: g * replace_zero(conj_gpu(x), 0.) / replace_zero(ans, 1.))
defvjp(add_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g: g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: g))
defvjp(conj_gpu,
       lambda ans, x: lambda g: conj_gpu(g))
defvjp(divide_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g:   g / y),
       lambda ans, x, y : unbroadcast_f(y, lambda g: -g * x / y ** 2))
defvjp(matmul_gpu, matmul_vjp_0, matmul_vjp_1)
defvjp(multiply_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g: y * g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: x * g))
defvjp(square_gpu,
       lambda ans, x : lambda g: g * 2 * x)
defvjp(subtract_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g: g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: -g))
defvjp(sum_gpu, grad_sum_gpu)
defvjp(swapaxes_gpu,
       lambda ans, x, axis1, axis2: lambda g: swapaxes_gpu(g, axis2, axis1))

### HELPER METHODS ###

"""
This function gets the .dtype attribute.
"""
dtype_gpu = lambda a_gpu: a_gpu.dtype


def grad_sum_gpu(ans, x, axis=None, keepdims=False, dtype=None):
    """
    This function is analogous to grad_np_sum in autograd.
    """
    shape, dtype = shape_gpu(x), dtype_gpu(x)

    return lambda g: repeat_to_match_shape(g, shape, dtype, axis, keepdims)[0]


def matmul_adjoint_0(B, G, A_meta, B_ndim):
    """
    This function is analogous to its autograd counterpart.
    """
    if ndim_gpu(G) == 0:  # A_ndim == B_ndim == 1
        return unbroadcast(G * B, A_meta)
    _, A_ndim, _, _ = A_meta
    if A_ndim == 1:
        G = expand_dims_gpu(G, ndim_gpu(G) - 1)
    if B_ndim == 1:  # The result we need is an outer product
        B = expand_dims_gpu(B, 0)
        G = expand_dims_gpu(G, ndim_gpu(G))
    else:  # We need to swap the last two axes of B
        B = swapaxes_gpu(B, B_ndim - 2, B_ndim - 1)
    result = matmul_gpu(G, B)

    return unbroadcast(result, A_meta)


def matmul_adjoint_1(A, G, A_ndim, B_meta):
    """
    This function is analogous to its autograd counterpart.
    """
    if ndim_gpu(G) == 0:  # A_ndim == B_ndim == 1
        return unbroadcast(G * A, B_meta)
    _, B_ndim, _, _ = B_meta
    B_is_vec = (B_ndim == 1)
    if B_is_vec:
        G = expand_dims_gpu(G, ndim_gpu(G))
    if A_ndim == 1:  # The result we need is an outer product
        A = expand_dims_gpu(A, 1)
        G = expand_dims_gpu(G, ndim_gpu(G) - 1)
    else:  # We need to swap the last two axes of A
        A = swapaxes_gpu(A, A_ndim - 2, A_ndim - 1)
    result = matmul_gpu(A, G)
    if B_is_vec:
        result = squeeze_gpu(result, ndim_gpu(G) - 1)

    return unbroadcast(result, B_meta)


def matmul_vjp_0(ans, A, B):
    """
    This function is analogous to its autograd counterpart.
    """
    A_meta = metadata(A)
    B_ndim = ndim_gpu(B)

    return lambda g: matmul_adjoint_0(B, g, A_meta, B_ndim)


def matmul_vjp_1(ans, A, B):
    """
    This function is analogous to its autograd counterpart.
    """
    A_ndim = ndim_gpu(A)
    B_meta = metadata(B)

    return lambda g: matmul_adjoint_1(A, g, A_ndim, B_meta)


def metadata(a_gpu):
    """
    This function behaves like autograd.numpy.metadata.
    """
    a_shape = a_gpu.shape
    a_ndim = a_gpu.ndim
    a_dtype = a_gpu.dtype
    a_iscomplex = iscomplexobj_gpu(a_gpu)
    a_metadata = (a_shape, a_ndim, a_dtype, a_iscomplex)

    return a_metadata


"""
This function gets the .ndim attribute.
"""
ndim_gpu = lambda a_gpu: a_gpu.ndim


def repeat_to_match_shape(g, shape, dtype, axis, keepdims):
    """
    This function is analogous to its autograd counterpart.
    """
    if shape == ():
      return g, 1
    axis = list(axis) if isinstance(axis, tuple) else axis
    new_shape = onp.array(shape)
    new_shape[axis] = 1
    num_reps = onp.prod(onp.array(shape)[axis])

    return reshape_gpu(g, new_shape) + zeros_gpu(shape, dtype=dtype), num_reps


def replace_zero(x, val):
    """
    This function is analogous to its autograd counterpart.
    """
    return where_gpu(x, x, val)


"""
This function gets the .shape attribute.
"""
shape_gpu = lambda a_gpu: a_gpu.shape


def unbroadcast(x, target_meta, broadcast_idx=0,):
    """
    This function is analogous to its autograd counterpart.
    """
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while ndim_gpu(x) > target_ndim:
        x = sum_gpu(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = sum_gpu(x, axis=axis, keepdims=True)
    if iscomplexobj_gpu(x) and not target_iscomplex:
        x = real_gpu(x)
    
    return x
            

def unbroadcast_f(target, f,):
    """
    This function is analgogous to its autograd counterpart.
    """
    target_meta = metadata(target)
    return lambda g: unbroadcast(f(g), target_meta)
