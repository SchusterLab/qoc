"""
pycuda_boxes.py - This module defines a class to wrap pycuda arrays.
"""

from autograd.extend import (Box, primitive,)
from pycuda.gpuarray import GPUArray

from qoc.standard.extensions.autograd_extensions.pycuda.pycuda_wrapper import (
    abs_gpu, add_gpu, divide_gpu, matmul_gpu, multiply_gpu,
    power_gpu, subtract_gpu, trace_gpu, transpose_gpu,
)

class GPUArrayBox(Box):
    """
    This class is defined analogously to its numpy counterpart:
    https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_boxes.py
    """
    
    __slots__ = []
    __array_priority__ = 100.0

    @primitive
    def __getitem__(A, idx): return A[idx]

    shape = property(lambda self: self._value.shape)
    ndim  = property(lambda self: self._value.ndim)
    size  = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: transpose_gpu(self))
    flags = property(lambda self: self._value.flags)
    get = property(lambda self: self._value.get)
    def __len__(self): return len(self._value)
    def astype(self, *args, **kwargs): return self._value.astype(*args, **kwargs)
    
    def __abs__(self): return abs_gpu(self)
    def __add__(self, other): return add_gpu(self, other)
    def __div__(self, other): return divide_gpu(  self, other)
    def __hash__(self): return id(self)
    def __matmul__(self, other): return matmul_gpu(self, other)
    def __mul__(self, other): return multiply_gpu(self, other)
    def __pow__(self, other): return power_gpu(self, other)
    def __radd__(self, other): return add_gpu(other, self)
    def __rsub__(self, other): return subtract_gpu(other, self)
    def __rmul__(self, other): return multiply_gpu(other, self)
    def __rdiv__(self, other): return divide_gpu(other, self)
    def __rmatmul__(self, other): return matmul_gpu(other, self)
    def __sub__(self, other): return subtract_gpu(self, other)

GPUArrayBox.register(GPUArray)
