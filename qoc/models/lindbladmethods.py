"""
lindbladmethods.py - This module contains math related
to the computation of the lindblad equation.
"""

from qoc.models.operationpolicy import (OperationPolicy,)
from qoc.standard import (commutator, conjugate_transpose,
                          matmuls,)

### MAIN METHODS ###

def get_lindbladian(densities, dissipators=None, hamiltonian=None,
                    operators=None,
                    operation_policy=OperationPolicy.CPU):
    """
    Compute the action of the lindblad equation on a single (set of)
    density matrix (matrices). This implementation uses the definiton:
    https://en.wikipedia.org/wiki/Lindbladian.

    Args:
    densities :: ndarray - the probability density matrices
    dissipators :: ndarray - the lindblad dissipators
    hamiltonian :: ndarray
    operators :: ndarray - the lindblad operators
    operation_policy :: qoc.OperationPolicy - how computations should be
        performed, e.g. CPU, GPU, sparse, etc.

    Returns:
    lindbladian :: ndarray - the lindbladian operator acting on the densities
    """
    if not (hamiltonian is None):
        lindbladian = -1j * commutator(hamiltonian, densities,
                                       operation_policy=operation_policy)
    else:
        lindbladian = 0
    if ((not (operators is None))
      and (not (dissipators is None))):
        operators_dagger = conjugate_transpose(operators,
                                               operation_policy=operation_policy)
        operators_product = matmuls(operators_dagger, operators,
                                    operation_policy=operation_policy)
        for i, operator in enumerate(operators):
            dissipator = dissipators[i]
            operator_dagger = operators_dagger[i]
            operator_product = operators_product[i]
            lindbladian = (lindbladian
                           + (dissipator
                              * (matmuls(operator, densities, operator_dagger,
                                         operation_policy=operation_policy)
                                 - 0.5 * matmuls(operator_product, densities,
                                                 operation_policy=operation_policy)
                                 - 0.5 * matmuls(densities, operator_product,
                                                 operation_policy=operation_policy))))
        #ENDFOR
    #ENDIF
    return lindbladian


### MODULE TESTS ###

def _test():
    import numpy as np
    
    # Test get_lindbladian on a hand verified solution.
    p = np.array(((1, 1), (1, 1)))
    ps = np.stack((p,))
    h = np.array(((0, 1), (1, 0)))
    g = 1
    gs = np.array((1,))
    l = np.array(((1, 0), (0, 0)))
    ls = np.stack((l,))
    lindbladian = get_lindbladian(p, gs, h, ls)
    expected_lindbladian = np.array(((0, -0.5),
                                     (-0.5, 0)))
    assert(np.allclose(lindbladian, expected_lindbladian))

if __name__ == "__main__":
    _test()
