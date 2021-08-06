
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
def krylov(HILBERT_SIZE):
    state = np.sqrt(1 / HILBERT_SIZE) * np.ones(HILBERT_SIZE)
    ANNIHILATION_OPERATOR = get_annihilation_operator(HILBERT_SIZE)
    CREATION_OPERATOR = get_creation_operator(HILBERT_SIZE)
    H0 = np.matmul(CREATION_OPERATOR, ANNIHILATION_OPERATOR) + 0.1 * (ANNIHILATION_OPERATOR + CREATION_OPERATOR)
    A = -1j * 0.1 * H0
    state=expm_multiply(A,state)
    return state
krylov(2000)