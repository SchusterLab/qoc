"""
definitions.py - This moudle defines a common namespace for functions
that act on pycuda arrays.

NOTE:
skcuda.misc functions are preferred over the pycuda.gpuarray.GPUArray object
methods because skcuda supports array broadcasting like numpy.
We want the gpu functions to have the same behavior as their cpu counterparts.
"""

from autograd.extend import primitive
import pycuda.cumath as cumath
import skcuda.linalg as culinalg
import skcuda.misc as cumisc

from qoc.standard.extensions.autograd_extensions.pycuda.pycuda_wrapper.custom_functions import (
    _abs_gpu, _power_gpu, _stack_gpu
)

GPU_FUNCTIONS = {
    "abs_gpu": _abs_gpu,
    "add_gpu": cumisc.add,
    "divide_gpu": cumisc.divide,
    "matmul_gpu": culinalg.dot,
    "multiply_gpu": cumisc.multiply,
    "power_gpu": _power_gpu,
    "stack_gpu": _stack_gpu,
    "subtract_gpu": cumisc.subtract,
    "sum_gpu": cumisc.sum,
    "trace_gpu": culinalg.trace,
    "transpose_gpu": culinalg.transpose,
}

def wrap_namespace(old, new):
    """
    This function is analagous to that in autograd.
    https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_wrapper.py
    """
    for name, obj in old.items():
        new[name] = primitive(obj)


wrap_namespace(GPU_FUNCTIONS, globals())
