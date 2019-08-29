"""
gpufunctions.py - This module provides a common namespace for all gpu functions.
Note that this is required because autograd can't differentiate object method calls.
"""

from autograd.extend import primitive
import skcuda.linalg as culinalg
import skcuda.misc as cumisc

@primitive
abs_gpu = lambda x: x.abs()

@primitve
add_gpu = lambda x, y: x + y

@primitive
conj_gpu = lambda x: x.conj()

@primitive
divide_gpu = cumisc.divide

@primitive
matmul_gpu = lambda x, y: x.dot(y)

@primitive
multiply_gpu = lambda x, y: x.__mul__(y)

@primitve
subtract_gpu = cumisc.subtract

@primitive
transpose_gpu = lambda x, **kwargs: x.transpose(**kwargs)





    

