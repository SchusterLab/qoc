"""
resolve_lindblad.py - Figure out why 1_transmon_pi_decoherence.py yields
a non-physical density matrix.
"""

import h5py as h5
import numpy as np
from qutip import (Qobj, mesolve,)

from qoc.standard import(SIGMA_Z, get_annihilation_operator,
                         get_creation_operator, SIGMA_PLUS)

FILE_PATH = "/home/tcpropson/repos/qoc/examples/out/00023_transmon_pi_decoherence.h5"

def main():
    # Fetch controls.
    h5file = h5.File(FILE_PATH)
    error = h5file["error"]
    error = np.where(error, error, np.finfo(np.float64).max)
    index = np.argmin(error)
    # final_density = h5file["densities"][index][0]
    controls = h5file["controls"][index]
    controls = np.append(controls, np.array([[0]]))
    controls_r = np.real(controls)
    controls_i = np.imag(controls)

    # Construct system.
    hilbert_size = 2
    a = get_annihilation_operator(hilbert_size)
    a_dagger = get_creation_operator(hilbert_size)
    h_0 = Qobj(SIGMA_Z / 2)
    h_r = Qobj(a + a_dagger)
    h_i = Qobj(1j * (a - a_dagger))
    rho0 = Qobj(np.array([[1, 0], [0, 0]]))
    c_ops = list()
    e_ops = list()
    evolution_time = control_step_count = 10
    tlist = np.linspace(0, evolution_time, control_step_count + 1)
    h = [h_0, [h_r, controls_r], [h_i, controls_i]]
    result = mesolve(h, rho0, tlist, c_ops, e_ops)
    final_density = result.states[-1]
    print(final_density)
    


if __name__ == "__main__":
    main()
