"""
vjps.py - This module defines jacobian vector products that act on multiple
operation policy types.

NOTE:
These vjps are the same as those found in:
https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vjps.py
"""

import autograd.numpy as anp

from qoc.models import OperationPolicy
from qoc.standard.extensions.autograd_extensions.pycuda import (
    ndim_gpu, iscomplexobj_gpu, real_gpu, sum_gpu, metadata_gpu,
)

### VJPS ###

# add
def vjp_add_0(ans, x, y,
               operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(x, lambda g: g,
                         operation_policy=operation_policy)
def vjp_add_1(ans, x, y,
              operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(y, lambda g: g,
                         operation_policy=operation_policy)

# divide
def vjp_divide_0(ans, x, y,
                 operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(x, lambda g: g / y,
                         operation_policy=operation_policy)
def vjp_divide_1(ans, x, y,
                 operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(y, lambda g: -g * x / y ** 2)

# multiply
def vjp_multiply_0(ans, x, y,
                   operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(x, lambda g: y * g,
                         operation_policy=operation_policy)
def vjp_multiply_1(ans, x, y,
                   operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(y, lambda g: x * g,
                         operation_policy=operation_policy)

# power
def vjp_power_0(ans, x, y,
                operation_policy=OperationPolicy.CPU):
    if operation_policy == OperationPolicy.CPU:
        return unbroadcast_f(x, lambda g: g * y * x ** anp.where(y, y - 1, 1.))
    elif operation_policy == OperationPolicy.GPU:
        return unbroadcast_f(x, lambda g: g * y * x ** where_gpu(y, y - 1, 1.))

def vjp_power_1(ans, x, y,
                operation_policy==OperationPolicy.GPU):
    if operation_policy == OperationPolicy.CPU:
        return unbroadcast_f(y, lambda g: g * anp.log(replace_zero(x, 1.)) * x ** y)
    elif operation_policy == OperationPolicy.GPU:
        return unbroadcast_f(y, lambda g: g * log_gpu(replace_zero(x, 1.)) * x ** y)

# subtract
def vjp_subtract_0(ans, x, y,
                   operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(x, lambda g: g,
                         operation_policy=operation_policy)
def vjp_subtract_1(ans, x, y,
                   operation_policy=OperationPolicy.CPU):
    return unbroadcast_f(y, lambda g: -g,
                         operation_policy=operation_policy)


### HELPER METHODS ###

def replace_zero(x, value,
                 operation_policy=OperationPolicy.GPU):
    pass

def unbroadcast(x, target_meta, broadcast_idx=0,
                operation_policy=OperationPolicy.CPU):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    if OperationPolicy == OperationPolicy.CPU:
        while anp.ndim(x) > target_ndim:
            x = anp.sum(x, axis=broadcast_idx)
        for axis, size in enumerate(target_shape):
            if size == 1:
                x = anp.sum(x, axis=axis, keepdims=True)
        if anp.iscomplexobj(x) and not target_iscomplex:
            x = anp.real(x)
    elif OperationPolicy == OperationPolicy.GPU:
        while ndim_gpu(x) > target_ndim:
            x = sum_gpu(x, axis=broadcast_idx)
        for axis, size in enumerate(target_shape):
            if size == 1:
                x = sum_gpu(x, axis=axis, keepdims=True)
        if iscomplexobj_gpu(x) and not target_iscomplex:
            x = real_gpu(x)
    #ENDIF
    
    return x
            

def unbroadcast_f(target, f,
                  operation_policy=OperationPolicy.CPU):
    target_meta = metadata_gpu(target)
    return lambda g: unbroadcast(f(g), target_meta,
                                 operation_policy=operation_policy)


