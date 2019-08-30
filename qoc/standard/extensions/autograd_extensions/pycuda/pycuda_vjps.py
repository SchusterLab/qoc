"""
pycuda_vjps.py - This module defines vector jacobian products for the functions
under the common namespace.

NOTE:
These definitions are analagous to their numpy counterparts:
https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py
"""

from qoc.standard.extensions.autograd_extensions.pycuda.pycuda_wrapper import (
    abs_gpu, add_gpu, divide_gpu, matmul_gpu, multiply_gpu, power_gpu,
    stack_gpu, subtract_gpu, sum_gpu, trace_gpu, transpose_gpu,
)

### VJPS ###

defvjp(abs_gpu,)
defvjp(add_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g: g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: g))
defvjp(divide_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g:   g / y),
       lambda ans, x, y : unbroadcast_f(y, lambda g: - g * x / y ** 2))
defvjp(matmul_gpu,)
defvjp(multiply_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g: y * g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: x * g))
defvjp(power_gpu,)
defvjp(subtract_gpu,
       lambda ans, x, y : unbroadcast_f(x, lambda g: g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: -g))
defvjp(sum_gpu,)
defvjp(trace_gpu,)
defvjp(transpose_gpu,)


### HELPER METHODS ###

def unbroadcast(x, target_meta, broadcast_idx=0,):
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
    target_meta = metadata_gpu(target)
    return lambda g: unbroadcast(f(g), target_meta)


