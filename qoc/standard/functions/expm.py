"""
expm.py - a module for all things e^M
"""

from autograd import jacobian
from autograd.extend import (defvjp as autograd_defvjp,
                             primitive as autograd_primitive)
import numpy as np
import scipy.linalg as la

from qoc.models.operationpolicy import OperationPolicy
from qoc.standard.constants import (get_eij)
from qoc.standard.functions import (conjugate_transpose,
                                    matmuls,
                                    mult_rows, mult_cols)
from qoc.standard.autograd_extensions import ans_jacobian

@autograd_primitive
def expm(matrix, operation_policy=OperationPolicy.CPU):
    """
    Compute the matrix exponential of a matrix.
    Args:
    matrix :: numpy.ndarray - the matrix to exponentiate
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method
    Returns:
    _expm :: numpy.ndarray - the exponentiated matrix
    """
    if operation_policy == OperationPolicy.CPU:
        _expm = la.expm(matrix)
    else:
        pass

    return _expm


def _expm_vjp(exp_matrix, matrix, operation_policy=OperationPolicy.CPU):
    """
    Construct the left-multiplying vector jacobian product function
    for the matrix exponential.

    Intuition:
    `dfinal_dexpm` is the jacobian of `final` with respect to each element `expmij`
    of `exp_matrix`. `final` is the output of the first function in the
    backward differentiation series. It is also the output of the last
    function evaluated in the chain of functions that is being differentiated,
    i.e. the final cost function. The goal of `vjp_function` is to take
    `dfinal_dexpm` and yield `dfinal_dmatrix` which is the jacobian of
    `final` with respect to each element `mij` of `matrix`.
    To compute the frechet derivative of the matrix exponential with respect
    to each element `mij`, we use the approximation that
    dexpm_dmij = np.matmul(Eij, exp_matrix). Since Eij has a specific
    structure we don't need to do the full matrix multiplication and instead
    use some indexing tricks.

    Args:
    exp_matrix :: numpy.ndarray - the matrix exponential of matrix
    matrix :: numpy.ndarray - the matrix that was exponentiated
    operation_policy :: qoc.OperationPolicy - what data type is
        used to perform the operation and with which method

    Returns:
    vjp_function :: numpy.ndarray -> numpy.ndarray - the function that takes
        the gradient of the final function, with respect to exp_matrix,
        to the gradient of the final function with respect to "matrix"
    """
    if operation_policy == OperationPolicy.CPU:
        matrix_size = matrix.shape[0]
        
        def _expm_vjp_cpu(dfinal_dexpm):
            dfinal_dmatrix = np.zeros((matrix_size, matrix_size), dtype=np.complex128)

            # Compute a first order approximation of the frechet derivative of the matrix
            # exponential in every unit direction Eij.
            for i in range(matrix_size):
                for j in range(matrix_size):
                    dexpm_dmij_rowi = exp_matrix[j,:]
                    dfinal_dmatrix[i, j] = np.sum(np.multiply(dfinal_dexpm[i, :], dexpm_dmij_rowi))
                #ENDFOR
            #ENDFOR

            return dfinal_dmatrix
        #ENDDEF

        vjp_function = _expm_vjp_cpu
    else:
        pass

    return vjp_function


autograd_defvjp(expm, _expm_vjp)


### DEXPM ###

PADE_ORDER_3 = 3
PADE_ORDER_5 = 5
PADE_ORDER_7 = 7
PADE_ORDER_9 = 9
PADE_ORDER_13 = 13

class ExpmFrechetNormalState(object):
    """
    NOTE: This code is deprecated in favor of an approximation to the frechet derivative.
    This code is kept here because I worked hard on it and do not want to remove it from
    the repository.

    This class encapsulates necessary information to obtain the
    frechet derivative of the matrix
    exponential at a normal matrix A in the direction of a unit matrix Eij.

    This implementation is adapted from scipy.linalg.expm_frechet.
    Please see SCIPY_LICENSE.txt in the same directory as this file for
    the copyright disclaimer.
    Scipy Implementation:
    https://github.com/scipy/scipy/blob/master/scipy/linalg/_expm_frechet.py
    Paper that describes implementation:
    https://epubs.siam.org/doi/abs/10.1137/080716426
    
    This implementation follows the naming conventions of the original code,
    and compromises on human readability for that reason.
    The goal of this implementation is to be able to compute every frechet derivative
    of the matrix exponential at the matrix A in each direction Eij,
    where both i,j in {0, ..., matrix_size - 1}, as fast as possible. This implementation
    makes use of the fact that A is a normal matrix which can be diagonalized
    with unitary matrices. Values that are used in every computation of expm_frechet
    are saved and reused.

    To use this class, pass the normal matrix A to the constructor. Then call
    the method: .expm_frechet_eij(i :: int, j :: int).
    """
    def __init__(self, A):
        ell_table_61 = (
            None,
            # 1
            2.11e-8,
            3.56e-4,
            1.08e-2,
            6.49e-2,
            2.00e-1,
            4.37e-1,
            7.83e-1,
            1.23e0,
            1.78e0,
            2.42e0,
            # 11
            3.13e0,
            3.90e0,
            4.74e0,
            5.63e0,
            6.56e0,
            7.52e0,
            8.53e0,
            9.56e0,
            1.06e1,
            1.17e1,
        )
        A_vec, p = la.eig(A)
        self. p = p
        self.p_dagger = conjugate_transpose(p)
        A_vec_norm_1 = la.norm(A_vec, 1)

        if A_vec_norm_1 <= ell_table_61[PADE_ORDER_3]:
            self.expm_frechet_eij = self.expm_frechet_eij_pade_3
            self.b = b = (120., 60., 12., 1.)
            self.A_vec = A_vec
            self.A_vec_2 = A_vec_2 = np.square(A_vec)
            U_vec = A_vec * (b[3] * A_vec_2 + b[1])
            V_vec = b[2] * A_vec_2 + b[0]
            self.lu_piv = la.lu_factor(-np.diag(U_vec) + np.diag(V_vec))
            self.R_vec = np.exp(A_vec)
        elif A_vec_norm_1 <= ell_table_61[PADE_ORDER_5]:
            self.expm_frechet_eij = self.expm_frechet_eij_pade_5
            self.b = b = (30240., 15120., 3360., 420., 30., 1.)
            self.A_vec = A_vec
            self.A_vec_2 = A_vec_2 = np.square(A_vec)
            self.A_vec_4 = A_vec_4 = np.square(A_vec_2)
            U_vec = A_vec * (b[5] * A_vec_4 + b[3] * A_vec_2 + b[1])
            V_vec = b[4] * A_vec_4 + b[2] * A_vec_2 + b[0]
            self.lu_piv = la.lu_factor(-np.diag(U_vec) + np.diag(V_vec))
            self.R_vec = np.exp(A_vec)
        elif A_vec_norm_1 <= ell_table_61[PADE_ORDER_7]:
            self.expm_frechet_eij = self.expm_frechet_eij_pade_7
            self.b = b = (17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.)
            self.A_vec = A_vec
            self.A_vec_2 = A_vec_2 = np.square(A_vec)
            self.A_vec_4 = A_vec_4 = np.square(A_vec_2)
            self.A_vec_6 = A_vec_6 = A_vec_4 * A_vec_2
            U_vec = A_vec * (b[7] * A_vec_6 + b[5] * A_vec_4 + b[3] * A_vec_2 + b[1])
            V_vec = b[6] * A_vec_6 + b[4] * A_vec_4 + b[2] * A_vec_2 + b[0]
            self.lu_piv = la.lu_factor(-np.diag(U_vec) + np.diag(V_vec))
            self.R_vec = np.exp(A_vec)
        elif A_vec_norm_1 <= ell_table_61[PADE_ORDER_9]:
            self.expm_frechet_eij = self.expm_frechet_eij_pade_9
            self.b = b = (17643225600., 8821612800., 2075673600., 302702400., 30270240.,
                          2162160., 110880., 3960., 90., 1.)
            self.A_vec = A_vec
            self.A_vec_2 = A_vec_2 = np.square(A_vec)
            self.A_vec_4 = A_vec_4 = np.square(A_vec_2)
            self.A_vec_6 = A_vec_6 = A_vec_2 * A_vec_4
            self.A_vec_8 = A_vec_8 = np.square(A_vec_4)
            U_vec = A_vec * (b[9] * A_vec_8 + b[7] * A_vec_6
                             + b[5] * A_vec_4 + b[3] * A_vec_2 + b[1])
            V_vec = b[8] * A_vec_8 + b[6] * A_vec_6 + b[4] * A_vec_4 + b[2] * A_vec_2 + b[0]
            self.lu_piv = la.lu_factor(-np.diag(U_vec) + np.diag(V_vec))
            self.R_vec = np.exp(A_vec)
        else:
            self.expm_frechet_eij = self.expm_frechet_eij_pade_13
            self.s = s = max(0, int(np.ceil(np.log2((A_vec_norm_1
                                                     / ell_table_61[PADE_ORDER_13])))))
            self.b = b = (64764752532480000., 32382376266240000., 7771770303897600.,
                          1187353796428800., 129060195264000., 10559470521600.,
                          670442572800., 33522128640., 1323241920., 40840800., 960960.,
                          16380., 182., 1.)
            A_vec = A_vec * 2.0**-s
            self.A_vec = A_vec
            self.A_vec_2 = A_vec_2 = np.square(A_vec)
            self.A_vec_4 = A_vec_4 = np.square(A_vec_2)
            self.A_vec_6 = A_vec_6 = A_vec_2 * A_vec_4
            self.W1_vec = W1_vec = b[13] * A_vec_6 + b[11] * A_vec_4 + b[9] * A_vec_2
            W2_vec = b[7] * A_vec_6 + b[5] * A_vec_4 + b[3] * A_vec_2 + b[1]
            self.Z1_vec = Z1_vec = b[12] * A_vec_6 + b[10] * A_vec_4 + b[8] * A_vec_2
            Z2_vec = b[6] * A_vec_6 + b[4] * A_vec_4 + b[2] * A_vec_2 + b[0]
            self.W_vec = W_vec = A_vec_6 * W1_vec + W2_vec
            U_vec = A_vec * W_vec
            V_vec = A_vec_6 * Z1_vec + Z2_vec
            self.lu_piv = la.lu_factor(-np.diag(U_vec) + np.diag(V_vec))
            self.R_vec = np.exp(A_vec)


    def expm_frechet_eij_pade_3(self, i, j):
        # Unpack.
        b = self.b
        p = self.p
        p_dagger = self.p_dagger
        A_vec = self.A_vec
        A_vec_2 = self.A_vec_2
        R_vec = self.R_vec

        # Compute.
        p_dagger_coli_matrix = np.repeat(np.vstack(p_dagger[:, i],),
                                      p_dagger.shape[0], axis=1)
        E = mult_cols(p_dagger_coli_matrix, p[j, :])
        M2 = mult_rows(E, A_vec) + mult_cols(E, A_vec)
        Lu = mult_rows(b[3] * M2, A_vec) + mult_cols(E, b[3] * A_vec_2 + b[1])
        Lv = b[2] * M2
        L = la.lu_solve(self.lu_piv, Lu + Lv + mult_cols(Lu - Lv, R_vec))

        return matmuls(p, L, p_dagger)


    def expm_frechet_eij_pade_5(self, i, j):
        # Unpack.
        b = self.b
        p = self.p
        p_dagger = self.p_dagger
        A_vec = self.A_vec
        A_vec_2 = self.A_vec_2
        A_vec_4 = self.A_vec_4
        R_vec = self.R_vec

        # Compute.
        p_dagger_coli_matrix = np.repeat(np.vstack(p_dagger[:, i],),
                                      p_dagger.shape[0], axis=1)
        E = mult_cols(p_dagger_coli_matrix, p[j, :])
        M2 = mult_rows(E, A_vec) + mult_cols(E, A_vec)
        M4 = mult_rows(M2, A_vec_2) + mult_cols(M2, A_vec_2)
        Lu = (mult_rows(b[5] * M4 + b[3] * M2, A_vec) +
              mult_cols(E, b[5] * A_vec_4 + b[3] * A_vec_2 + b[1]))
        Lv = b[4] * M4 + b[2] * M2
        L = la.lu_solve(self.lu_piv, Lu + Lv + mult_cols(Lu - Lv, R_vec))

        return matmuls(p, L, p_dagger)


    def expm_frechet_eij_pade_7(self, i, j):
        # Unpack.
        b = self.b
        p = self.p
        p_dagger = self.p_dagger
        A_vec = self.A_vec
        A_vec_2 = self.A_vec_2
        A_vec_4 = self.A_vec_4
        A_vec_6 = self.A_vec_6
        R_vec = self.R_vec

        # Compute.
        p_dagger_coli_matrix = np.repeat(np.vstack(p_dagger[:, i],),
                                      p_dagger.shape[0], axis=1)
        E = mult_cols(p_dagger_coli_matrix, p[j, :])
        M2 = mult_rows(E, A_vec) + mult_cols(E, A_vec)
        M4 = mult_rows(M2, A_vec_2) + mult_cols(M2, A_vec_2)
        M6 = mult_rows(M2, A_vec_4) + mult_cols(M4, A_vec_2)
        Lu = (mult_rows(b[7] * M6 + b[5] * M4 + b[3] * M2, A_vec) +
              mult_cols(E, b[7] * A_vec_6 + b[5] * A_vec_4 + b[3] * A_vec_2 + b[1]))
        Lv = b[6] * M6 + b[4] * M4 + b[2] * M2
        L = la.lu_solve(self.lu_piv, Lu + Lv + mult_cols(Lu - Lv, R_vec))

        return matmuls(p, L, p_dagger)


    def expm_frechet_eij_pade_9(self, i, j):
        # Unpack.
        b = self.b
        p = self.p
        p_dagger = self.p_dagger
        A_vec = self.A_vec
        A_vec_2 = self.A_vec_2
        A_vec_4 = self.A_vec_4
        A_vec_6 = self.A_vec_6
        A_vec_8 = self.A_vec_8
        R_vec = self.R_vec

        # Compute.
        p_dagger_coli_matrix = np.repeat(np.vstack(p_dagger[:, i],),
                                      p_dagger.shape[0], axis=1)
        E = mult_cols(p_dagger_coli_matrix, p[j, :])
        M2 = mult_rows(E, A_vec) + mult_cols(E, A_vec)
        M4 = mult_rows(M2, A_vec_2) + mult_cols(M2, A_vec_2)
        M6 = mult_rows(M2, A_vec_4) + mult_cols(M4, A_vec_2)
        M8 = mult_rows(M4, A_vec_4) + mult_cols(M4, A_vec_4)
        Lu = (mult_rows(b[9] * M8 + b[7] * M6 + b[5] * M4 + b[3] * M2, A_vec) +
              mult_cols(E, (b[9] * A_vec_8 + b[7] * A_vec_6 + b[5] * A_vec_4
                            + b[3] * A_vec_2 + b[1])))
        Lv = b[8] * M8 + b[6] * M6 + b[4] * M4 + b[2] * M2
        L = la.lu_solve(self.lu_piv, Lu + Lv + mult_cols(Lu - Lv, R_vec))

        return matmuls(p, L, p_dagger)


    def expm_frechet_eij_pade_13(self, i, j):
        # Unpack.
        b = self.b
        p = self.p
        s = self.s
        p_dagger = self.p_dagger
        A_vec = self.A_vec
        A_vec_2 = self.A_vec_2
        A_vec_4 = self.A_vec_4
        A_vec_6 = self.A_vec_6
        W1_vec = self.W1_vec
        Z1_vec = self.Z1_vec
        W_vec = self.W_vec
        R_vec = self.R_vec

        # Compute.
        p_dagger_coli_matrix = np.repeat(np.vstack(p_dagger[:, i],),
                                      p_dagger.shape[0], axis=1)
        E = mult_cols(p_dagger_coli_matrix, p[j, :]) * 2.0 **-s
        M2 = mult_rows(E, A_vec) + mult_cols(E, A_vec)
        M4 = mult_rows(M2, A_vec_2) + mult_cols(M2, A_vec_2)
        M6 = mult_rows(M2, A_vec_4) + mult_cols(M4, A_vec_2)
        Lw1 = b[13] * M6 + b[11] * M4 + b[9] * M2
        Lw2 = b[7] * M6 + b[5] * M4 + b[3] * M2
        Lz1 = b[12] * M6 + b[10] * M4 + b[8] * M2
        Lz2 = b[6] * M6 + b[4] * M4 + b[2] * M2
        Lw = mult_rows(Lw1, A_vec_6) + mult_cols(M6, W1_vec) + Lw2
        Lu = mult_rows(Lw, A_vec) + mult_cols(E, W_vec)
        Lv = mult_rows(Lz1, A_vec_6) + mult_cols(M6, Z1_vec) + Lz2
        L = la.lu_solve(self.lu_piv, Lu + Lv + mult_cols(Lu - Lv, R_vec))

        # Square.
        for k in range(s):
            L = mult_rows(L, R_vec) + mult_cols(L, R_vec)
            R_vec = np.square(R_vec)

        return matmuls(p, L, p_dagger)


### MODULE TESTS ###

_BIG = 30
_TEST_DEPRECATED = False

def _get_skew_hermitian_matrix(matrix_size):
    matrix = (np.random.rand(matrix_size, matrix_size)
              + 1j * np.random.rand(matrix_size, matrix_size))
    
    return np.divide(matrix - conjugate_transpose(matrix), 2)

def _tests():
    """
    Run tests on the module.
    Args: none
    Returns: nothing
    """

    # Test that the end-to-end gradient of the matrix exponential is working.
    m = np.array([[1., 0.],
                  [0., 1.]])
    m_len = m.shape[0]
    dexpm_dm_expected = np.zeros((m_len, m_len, m_len, m_len), dtype=m.dtype)
    dexpm_dm_expected[0, 0, 0, 0] = np.exp(m)[0, 0]
    dexpm_dm_expected[0, 1, 0, 1] = np.exp(m)[0, 0]
    dexpm_dm_expected[1, 0, 1, 0] = np.exp(m)[1, 1]
    dexpm_dm_expected[1, 1, 1, 1] = np.exp(m)[1, 1]
    
    dexpm_dm = jacobian(expm, 0)(m)

    assert(np.allclose(dexpm_dm, dexpm_dm_expected))

    # These functions are not currently used by the package and do not
    # need to be tested.
    if _TEST_DEPRECATED:
        # Test that the frechet derivative of the normal matrix exponential
        # matches scipy's implementation.
        # A random scale factor is introduced to ensure that all pades get
        # evaluated.
        for matrix_size in range(2, _BIG):
            scale = np.random.randint(-7, 7)
            shm = _get_skew_hermitian_matrix(matrix_size) * 10 ** scale
            dexpm_state = ExpmFrechetNormalState(shm)
            for i in range(matrix_size):
                for j in range(matrix_size):
                    i = np.random.randint(0, matrix_size)
                    j = np.random.randint(0, matrix_size)
                    eij = get_eij(i, j, matrix_size)

                    expm_frechet_normal = dexpm_state.expm_frechet_eij(i, j)
                    expm_frechet = la.expm_frechet(shm, eij, compute_expm=False)

                    assert(np.allclose(expm_frechet_normal, expm_frechet))
                #ENDFOR
            #ENDFOR
        #ENDFOR


if __name__ == "__main__":
    _tests()
