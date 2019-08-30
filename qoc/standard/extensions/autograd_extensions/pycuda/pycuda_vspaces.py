"""
pycuda_vspaces.py - This module defines a class to facilitate
vector space interfacing with pycuda arrays.
"""

from autograd.extend import (VSpace,)
from pycuda.gpuarray import GPUArray

class GPUArrayVSpace(VSpace):
    """
    This class is defined analogously to its numpy counterpart:
    https://github.com/HIPS/autograd/blob/master/autograd/numpy/numpy_vspaces.py
    """
    
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
