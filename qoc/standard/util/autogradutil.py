"""
autogutil.py - This module provides utilities for interfacing with autograd.
"""

from autograd.core import make_vjp as _make_vjp
from autograd.extend import (primitive, Box, VSpace, vspace)
from autograd.wrap_util import (unary_to_nary,)
import numpy as np
from pycuda.gpuarray import GPUArray
from pycuda import cumath
import skcuda.linalg as culinalg
import skcuda.misc as cumisc

from qoc.standard.functions.convenience import stack_gpu

### DIFFERENTIAL OPERATORS ###

# This differential operator follows autograd's jacobian implementation.
# https://github.com/HIPS/autograd/blob/master/autograd/differential_operators.py
@unary_to_nary
def ans_jacobian(function, argnum):
    """
    Get the value and the jacobian of a function.
    This differential operator supports numpy and pycuda arrays.

    Args:
    function :: any -> any - the function to differentiate
    argnum :: int - the argument number to differentiate with respect to

    Returns:
    ans_jacobian any -> tuple(any :: any, jacobian :: ndarray) - a function
        that returns the value of `function` and the jacobian
        of `function` evaluated at a given argument of `function`
    """
    vjp, ans = _make_vjp(function, argnum)
    ans_vspace = vspace(ans)
    jacobian_shape = ans_vspace.shape + vspace(argnum).shape
    grads = list(map(vjp, ans_vspace.standard_basis()))
    if isinstance(grads[0], np.ndarray):
        jacobian = np.reshape(np.stack(grads), jacobian_shape)
    elif isinstance(grads[0], GPUArray):
        jacobian =  stack_gpu(grads).reshape(jacobian_shape)
    
    return ans, jacobian


### GPU EXTENSION ###

# These classes are defined analogously to those autograd
# defines for numpy arrays.
# https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_boxes.py
# https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vspaces.py

class GPUArrayBox(Box):
    __slots__ = []
    __array_priority__ = 100.0

    @primitive
    def __getitem__(A, idx): return A[idx]

    shape = property(lambda self: self._value.shape)
    ndim  = property(lambda self: self._value.ndim)
    size  = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: culinalg.transpose(self))
    flags = property(lambda self: self._value.flags)
    get = property(lambda self: self._value.get)
    def __len__(self): return len(self._value)
    def astype(self, *args, **kwargs): return self._value.astype(*args, **kwargs)

    def __add__(self, other): return cumisc.add(self, other)
    def __sub__(self, other): return cumisc.subtract(self, other)
    def __mul__(self, other): return cumisc.multiply(self, other)
    def __div__(self, other): return cumisc.divide(  self, other)
    def __matmul__(self, other): return culinalg.dot(self, other)
    def __radd__(self, other): return cumisc.add(other, self)
    def __rsub__(self, other): return cumisc.subtract(other, self)
    def __rmul__(self, other): return cumisc.multiply(other, self)
    def __rdiv__(self, other): return cumisc.divide(other, self)
    def __rmatmul__(self, other): return culinalg.dot(other, self)
    def __abs__(self): return cumath.fabs(self) # assumes floating point datatype
    def __hash__(self): return id(self)


GPUArrayBox.register(GPUArray)


class GPUArrayVSpace(VSpace):
    def __init__(self, value):
        if not isinstance(value, GPUArray):
            value = GPUArray.to_gpu(value)
        self.shape = value.shape
        self.dtype = value.dtype

    @property
    def size(self): return np.prod(self.shape)
    @property
    def ndim(self): return len(self.shape)
    def zeros(self): return pycuda.gpuarray.zeros(self.shape, self.dtype)
    def ones(self): return pycuda.gpuarray.to_gpu(np.ones(self.shape, dtype=self.dtype))

    def standard_basis(self):
      for idxs in np.ndindex(*self.shape):
          vect = pycuda.gpuarray.zeros(self.shape, self.dtype)
          vect[idxs] = np.array(1).astype(vect.dtype)
          yield vect

    def randn(self):
        return pycuda.gpuarray.gen_normal(self.shape, self.dtype)

    def _inner_prod(self, x, y):
        return pycuda.gpuarray.dot(x.ravel(), y.ravel())

class ComplexGPUArrayVSpace(GPUArrayVSpace):
    iscomplex = True

    @property
    def size(self): return np.prod(self.shape) * 2

    def ones(self):
        return pycuda.gpuarray.to_gpu(np.ones(self.shape, dtype=self.dtype)
                                      + 1.0j * np.ones(self.shape, dtype=self.dtype))

    def standard_basis(self):
      for idxs in np.ndindex(*self.shape):
          for v in [1.0, 1.0j]:
              vect = pycuda.gpuarray.zeros(self.shape, self.dtype)
              vect[idxs] = np.array(v).astype(vect.dtype)
              yield vect

    def randn(self):
        _randn = (np.array(np.random.randn(*self.shape)).astype(self.dtype)
                  + 1.0j * np.array(np.random.randn(*self.shape)).astype(self.dtype))
        return pycuda.gpuarray.to_gpu(_randn)

    def _inner_prod(self, x, y):
        return pycuda.gpuarray.dot(x.ravel().conj(), y.ravel()).real

    def _covector(self, x):
        return x.conj()

    
VSpace.register(GPUArray,
                lambda x: ComplexGPUArrayVSpace(x)
                if cumisc.iscomplextype(x.dtype)
                else GPUArrayVSpace(x))


### GPU VJPS ###

GPU_DIFFERENTIABLE_FUNCTIONS = (
    # pycuda.cumath
    cumath.fabs,
    
    # skcuda.linalg
    culinalg.dot,
    culinalg.transpose,

    # skcuda.misc
    cumisc.add,
    cumisc.divide,
    cumisc.multiply,
    cumisc.subtract,
)

# Register differentiable functions.
for function in GPU_DIFFERENTIABLE_FUNCTIONS:
    primitive(function)

# Define vjps.

def unbroadcast_gpu(x, target_meta, broadcast_idx=0):
    target_shape, target_ndim, dtype, target_iscomplex = target_meta
    while x.ndim > target_ndim:
        x = cumisc.sum(x, axis=broadcast_idx)
    for axis, size in enumerate(target_shape):
        if size == 1:
            x = cumisc.sum(x, axis=axis, keepdims=True)
    if cumisc.iscomplextype(x.dtype) and not target_iscomplex:
        x = x.real
    return x

def unbroadcast_f_gpu(target, f):
    target_shape = target.shape
    target_ndim = target.ndim
    target_dtype = target.dtype
    target_iscomplex = cumisc.iscomplextype(target_dtype)
    target_meta = (target_shape, target_ndim, target_dtype,
                   target_iscomplex)
    return lambda g: unbroadcast(f(g), target_meta)


defvjp(cumisc.add,
       lambda ans, x, y : unbroadcast_f(x, lambda g: g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: g))
defvjp(cumisc.multiply,
       lambda ans, x, y : unbroadcast_f(x, lambda g: y * g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: x * g))
defvjp(cumisc.subtract,
       lambda ans, x, y : unbroadcast_f(x, lambda g: g),
       lambda ans, x, y : unbroadcast_f(y, lambda g: -g))
defvjp(cumisc.divide,
       lambda ans, x, y : unbroadcast_f(x, lambda g:   g / y),
       lambda ans, x, y : unbroadcast_f(y, lambda g: - g * x / y**2))


