"""
resolve_lindblad.py - Figure out why 1_transmon_pi_decoherence.py yields
a non-physical density matrix.
"""

import os

import autograd.numpy as anp
import h5py as h5
import numpy as np
from qutip import (Qobj, mesolve,)

from qoc import evolve_lindblad_discrete
from qoc.standard import(SIGMA_Z, get_annihilation_operator,
                         get_creation_operator, SIGMA_PLUS, conjugate_transpose)

QOC_PATH = os.environ["QOC_PATH"]
FILE_PATH = os.path.join(QOC_PATH, "examples/out/00025_transmon_pi_decoherence.h5")

def main():
    # Fetch controls.
    h5file = h5.File(FILE_PATH)
    error = h5file["error"]
    error = np.where(error, error, np.finfo(np.float64).max)
    index = np.argmin(error)
    # final_density = h5file["densities"][index][0]
    controls = h5file["controls"][index]
    controls_qutip = np.vstack((controls, np.array([[0]])))
    controls_r_qutip = np.real(controls_qutip)
    controls_i_qutip = np.imag(controls_qutip)

    # Construct system.
    hilbert_size = 2
    a = get_annihilation_operator(hilbert_size)
    a_dagger = get_creation_operator(hilbert_size)
    h_0 = SIGMA_Z / 2
    h_r = a + a_dagger
    h_i = 1j * (a - a_dagger)
    rho0 = np.array([[1, 0], [0, 0]])
    evolution_time = control_step_count = 10

    # qoc
    hamiltonian = (lambda controls, time:
                   h_0
                   + controls[0] * a
                   + anp.conjugate(controls[0]) * a_dagger)
    # for i in range(control_step_count):
    #     hi = hamiltonian(controls[i], None)
    #     print("qoc_h[{}]:\n{}"
    #           "".format(i, hamiltonian(controls[i], None)))
    #     assert(np.allclose(hi, conjugate_transpose(hi)))
    #ENDFOR
    initial_density_0 = np.array([[1, 0], [0, 0]])
    initial_densities = np.stack((initial_density_0,))
    result = evolve_lindblad_discrete(control_step_count, evolution_time, initial_densities,
                                      controls=controls, hamiltonian=hamiltonian)
    final_density = result.final_densities[0]
    print("final_density_qoc:\n{}"
          "".format(final_density))

    # qutip
    h_0_qutip = Qobj(h_0)
    h_r_qutip = Qobj(h_r)
    h_i_qutip = Qobj(h_i)
    rho0_qutip = Qobj(rho0)
    c_ops = list()
    e_ops = list()
    tlist = np.linspace(0, evolution_time, control_step_count + 1)
    h = [h_0_qutip, [h_r_qutip, controls_r_qutip], [h_i_qutip, controls_i_qutip]]
    result_qutip = mesolve(h, rho0_qutip, tlist, c_ops, e_ops)
    final_density_qutip = result_qutip.states[-1]
    # print("qutip densities:\n{}"
    #       "".format(result_qutip.states))
    print("final_density_qutip:\n{}"
          "".format(final_density_qutip))

    # for i in range(len(tlist)):
    #     hi = h_0 + controls_r_qutip[i][0] * h_r + controls_i_qutip[i][0] * h_i
    #     print("qutip_h[{}]:\n{}"
    #           "".format(i, hi))
    # #ENDFOR


if __name__ == "__main__":
    main()
