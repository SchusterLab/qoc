"""
definitions.py - This moudle defines a common namespace for functions
that act on pycuda arrays.

NOTE:
skcuda functions are preferred over the pycuda.gpuarray.GPUArray object
methods because skcuda supports array broadcasting like numpy.
We want the gpu functions to have the same behavior as their cpu counterparts.
"""

from autograd.extend import primitive
import pycuda.cumath as cumath
import skcuda.linalg as culinalg
import skcuda.misc as cumisc

from qoc.autograd_extensions.pycuda.pycuda_wrapper.custom_functions import (
    _abs_gpu, _imag_gpu, _iscomplexobj_gpu, _power_gpu, _real_gpu,
    _reshape_gpu, _square_gpu, _stack_gpu, _swapaxes_gpu, _where_gpu,
)

GPU_FUNCTIONS = {
    "abs_gpu": _abs_gpu,
    "add_gpu": cumisc.add,
    "cos_gpu": cumath.cos,
    "conj_gpu": culinalg.conj,
    "divide_gpu": cumisc.divide,
    "exp_gpu": cumath.exp,
    "expand_dims_gpu": _expand_dims_gpu,
    "imag_gpu": _imag_gpu,
    "iscomplexobj_gpu": _iscomplexobj_gpu,
    "log_gpu": cumath.log,
    "log10_gpu": cumath.log10,
    "matmul_gpu": culinalg.dot,
    "max_gpu": cumisc.max,
    "min_gpu": cumisc.min,
    "multiply_gpu": cumisc.multiply,
    "ones_like_gpu": pycuda.gpuarray.ones_like,
    "power_gpu": _power_gpu,
    "real_gpu": _real_gpu,
    "reshape_gpu": _rehsape_gpu,
    "sin_gpu": cumath.sin,
    "sqrt_gpu": cumath.sqrt,
    "square_gpu": _square_gpu,
    "squeeze_gpu": _squeeze_gpu,
    "stack_gpu": _stack_gpu,
    "subtract_gpu": cumisc.subtract,
    "sum_gpu": cumisc.sum,
    "swapaxes_gpu": _swapaxes_gpu,
    "tan_gpu": cumath.tan,
    "trace_gpu": culinalg.trace,
    "where_gpu": _where_gpu,
    "zeros_gpu": pycuda.gpuarray.zeros,
}

def wrap_namespace(old, new):
    """
    This function is analagous to that in autograd.
    https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_wrapper.py
    """
    for name, obj in old.items():
        new[name] = primitive(obj)

wrap_namespace(GPU_FUNCTIONS, globals())
