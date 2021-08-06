
import numpy as np
from qoc.standard import (
                          get_annihilation_operator,
                          get_creation_operator,
                          )
import sys
from qoc.standard.functions.expm_manual import expm_pade
from scipy.sparse.linalg import expm_multiply,expm
from scipy.sparse import dia_matrix,bsr_matrix
from qoc.core.common import expm_frechet
def krylov_sparse(HILBERT_SIZE):
    state = np.sqrt(1 / HILBERT_SIZE) * np.ones(HILBERT_SIZE)
    diagnol=np.arange(HILBERT_SIZE)
    up_diagnol=np.sqrt(diagnol)
    low_diagnol=np.sqrt(np.arange(1,HILBERT_SIZE+1))
    data=[low_diagnol,diagnol,up_diagnol]
    offsets=[-1,0,1]
    A=-1j*0.1*dia_matrix((data, offsets),shape=(HILBERT_SIZE,HILBERT_SIZE))
    state=expm_multiply(A,state)
    return state
krylov_sparse(1000)