"""
test_standard.py - This module provides unit tests on the qoc.standard module.
"""

### qoc.standard.constants ###

def test_constants():
    import numpy as np

    from qoc.standard.constants import (get_creation_operator,
                                        get_annihilation_operator)
    
    big = 100
    
    # Use the fact that (create)(annihilate) is the number operator
    # to test the creation and annihilation operator methods.
    for i in range(1, big):
        analytic_number_operator = np.diag(np.arange(i))
        generated_number_operator = np.matmul(get_creation_operator(i), get_annihilation_operator(i))
        assert np.allclose(generated_number_operator, analytic_number_operator)


### qoc.standard.costs ###

# TODO: implement me
def test_controlarea():
    pass


# TODO: implement me
def test_controlnorm():
    pass


# TODO: implement me
def test_controlvariation():
    pass


def test_forbiddensities():
    import numpy as np

    from qoc.standard import conjugate_transpose
    from qoc.standard.costs.forbiddensities import ForbidDensities
    
    system_eval_count = 11
    state0 = np.array([[1], [0]])
    density0 = np.matmul(state0, conjugate_transpose(state0))
    forbid0_0 = np.array([[1], [0]])
    density0_0 = np.matmul(forbid0_0, conjugate_transpose(forbid0_0))
    forbid0_1 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    density0_1 = np.matmul(forbid0_1, conjugate_transpose(forbid0_1))
    state1 = np.array([[0], [1]])
    density1 = np.matmul(state1, conjugate_transpose(state1))
    forbid1_0 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    density1_0 = np.matmul(forbid1_0, conjugate_transpose(forbid1_0))
    forbid1_1 = np.divide(np.array([[1j], [1j]]), np.sqrt(2))
    density1_1 = np.matmul(forbid1_1, conjugate_transpose(forbid1_1))
    densities = np.stack((density0, density1,))
    forbidden_densities0 = np.stack((density0_0, density0_1,))
    forbidden_densities1 = np.stack((density1_0, density1_1,))
    forbidden_densities = np.stack((forbidden_densities0, forbidden_densities1,))
    fd = ForbidDensities(forbidden_densities, system_eval_count)
    
    cost = fd.cost(None, densities, None)
    expected_cost = 7 / 640
    assert(np.allclose(cost, expected_cost,))


def test_forbidstates():
    import numpy as np

    from qoc.standard.costs.forbidstates import ForbidStates

    system_eval_count = 11
    state0 = np.array([[1], [0]])
    forbid0_0 = np.array([[1], [0]])
    forbid0_1 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    state1 = np.array([[0], [1]])
    forbid1_0 = np.divide(np.array([[1], [1]]), np.sqrt(2))
    forbid1_1 = np.divide(np.array([[1j], [1j]]), np.sqrt(2))
    states = np.stack((state0, state1,))
    forbidden_states0 = np.stack((forbid0_0, forbid0_1,))
    forbidden_states1 = np.stack((forbid1_0, forbid1_1,))
    forbidden_states = np.stack((forbidden_states0, forbidden_states1,))
    fs = ForbidStates(forbidden_states, system_eval_count)
    
    cost = fs.cost(None, states, None)
    expected_cost = np.divide(5, 80)
    assert(np.allclose(cost, expected_cost,))

    
def test_targetdensityinfidelity():
    import numpy as np

    from qoc.standard import conjugate_transpose
    from qoc.standard.costs.targetdensityinfidelity import TargetDensityInfidelity
    
    state0 = np.array([[0], [1]])
    density0 = np.matmul(state0, conjugate_transpose(state0))
    target_state0 = np.array([[1], [0]])
    target_density0 = np.matmul(target_state0, conjugate_transpose(target_state0))
    densities = np.stack((density0,), axis=0)
    targets = np.stack((target_density0,), axis=0)
    ti = TargetDensityInfidelity(targets)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 1))

    ti = TargetDensityInfidelity(densities)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 0.5))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    density0 = np.matmul(state0, conjugate_transpose(state0))
    density1 = np.matmul(state1, conjugate_transpose(state1))
    target_state0 = np.array([[1j], [0]])
    target_state1 = np.array([[1], [0]])
    target_density0 = np.matmul(target_state0, conjugate_transpose(target_state0))
    target_density1 = np.matmul(target_state1, conjugate_transpose(target_state1))
    densities = np.stack((density0, density1,), axis=0)
    targets = np.stack((target_density0, target_density1,), axis=0)
    ti = TargetDensityInfidelity(targets)
    cost = ti.cost(None, densities, None)
    expected_cost = 0.625
    assert(np.allclose(cost, expected_cost))

    
def test_targetdensityinfidelitytime():
    import numpy as np

    from qoc.standard import conjugate_transpose
    from qoc.standard.costs.targetdensityinfidelitytime import TargetDensityInfidelityTime

    system_eval_count = 11
    state0 = np.array([[0], [1]])
    density0 = np.matmul(state0, conjugate_transpose(state0))
    target_state0 = np.array([[1], [0]])
    target_density0 = np.matmul(target_state0, conjugate_transpose(target_state0))
    densities = np.stack((density0,), axis=0)
    targets = np.stack((target_density0,), axis=0)
    ti = TargetDensityInfidelityTime(system_eval_count, targets)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 0.1))

    ti = TargetDensityInfidelityTime(system_eval_count, densities)
    cost = ti.cost(None, densities, None)
    assert(np.allclose(cost, 0.05))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    density0 = np.matmul(state0, conjugate_transpose(state0))
    density1 = np.matmul(state1, conjugate_transpose(state1))
    target_state0 = np.array([[1j], [0]])
    target_state1 = np.array([[1], [0]])
    target_density0 = np.matmul(target_state0, conjugate_transpose(target_state0))
    target_density1 = np.matmul(target_state1, conjugate_transpose(target_state1))
    densities = np.stack((density0, density1,), axis=0)
    targets = np.stack((target_density0, target_density1,), axis=0)
    ti = TargetDensityInfidelityTime(system_eval_count, targets)
    cost = ti.cost(None, densities, None)
    expected_cost = 0.0625
    assert(np.allclose(cost, expected_cost))

    
def test_targetstateinfidelity():
    import numpy as np

    from qoc.standard.costs.targetstateinfidelity import TargetStateInfidelity

    state0 = np.array([[0], [1]])
    target0 = np.array([[1], [0]])
    states = np.stack((state0,), axis=0)
    targets = np.stack((target0,), axis=0)
    ti = TargetStateInfidelity(targets)
    cost = ti.cost(None, states, None)
    assert(np.allclose(cost, 1))

    ti = TargetStateInfidelity(states)
    cost = ti.cost(None, states, None)
    assert(np.allclose(cost, 0))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    target0 = np.array([[1j], [0]])
    target1 = np.array([[1], [0]])
    states = np.stack((state0, state1,), axis=0)
    targets = np.stack((target0, target1,), axis=0)
    ti = TargetStateInfidelity(targets)
    cost = ti.cost(None, states, None)
    assert(np.allclose(cost, .25))


def test_targetstateinfidelitytime():
    import numpy as np

    from qoc.standard.costs.targetstateinfidelitytime import TargetStateInfidelityTime

    system_eval_count = 11
    state0 = np.array([[0], [1]])
    target0 = np.array([[1], [0]])
    states = np.stack((state0,), axis=0)
    targets = np.stack((target0,), axis=0)
    ti = TargetStateInfidelityTime(system_eval_count, targets)
    cost = ti.cost(None, states, None)
    expected_cost = 0.1
    assert(np.allclose(cost, expected_cost))

    ti = TargetStateInfidelityTime(system_eval_count, states)
    cost = ti.cost(None, states, None)
    expected_cost = 0
    assert(np.allclose(cost, expected_cost))

    state0 = np.array([[1], [0]])
    state1 = (np.array([[1j], [1]]) / np.sqrt(2))
    target0 = np.array([[1j], [0]])
    target1 = np.array([[1], [0]])
    states = np.stack((state0, state1,), axis=0)
    targets = np.stack((target0, target1,), axis=0)
    ti = TargetStateInfidelityTime(system_eval_count, targets)
    cost = ti.cost(None, states, None)
    expected_cost = 0.025
    assert(np.allclose(cost, expected_cost))


### qoc.standard.functions ###

def test_expm():
    from autograd import jacobian
    import numpy as np

    from qoc.standard.functions.expm import expm
    
    # Test that the end-to-end gradient of the matrix exponential is working.
    m = np.array([[1., 0.],
                  [0., 1.]])
    m_len = m.shape[0]
    exp_m = np.exp(m)
    dexpm_dm_expected = np.zeros((m_len, m_len, m_len, m_len), dtype=m.dtype)
    dexpm_dm_expected[0, 0, 0, 0] = exp_m[0, 0]
    dexpm_dm_expected[0, 1, 0, 1] = exp_m[0, 0]
    dexpm_dm_expected[1, 0, 1, 0] = exp_m[1, 1]
    dexpm_dm_expected[1, 1, 1, 1] = exp_m[1, 1]
    
    dexpm_dm = jacobian(expm, 0)(m)

    assert(np.allclose(dexpm_dm, dexpm_dm_expected))


### qoc.standard.optimizers ###

def test_adam():
    import numpy as np
    
    from qoc.core.common import (strip_controls, slap_controls)
    from qoc.models.dummy import Dummy
    from qoc.standard.optimizers.adam import Adam
    
    # Check that the update method was implemented correctly
    # using hand-checked values.
    adam = Adam()
    grads = np.array([[0, 1],
                      [2, 3]])
    params = np.array([[0, 1],
                       [2, 3]], dtype=np.float64)
    params1 = np.array([[0,         0.999],
                        [1.999, 2.999]])
    params2 = np.array([[0,          0.99900003],
                        [1.99900001, 2.99900001]])
    
    adam.run(None, 0, params, None, None)
    params1_test = adam.update(params, grads)
    params2_test = adam.update(params1, grads)
    
    assert(np.allclose(params1_test, params1))
    assert(np.allclose(params2_test, params2))

    # Check that complex mapping works and params
    # without gradients are unaffected.
    gstate = Dummy()
    gstate.complex_controls = True
    grads = np.array([[1+1j, 0+0j],
                      [0+0j, -1-1j]])
    params = np.array([[1+2j, 3+4j],
                       [5+6j, 7+8j]])
    gstate.controls_shape = params.shape
    gstate.max_param_norms = np.ones(gstate.controls_shape[0]) * 10
    
    flat_controls = strip_controls(gstate.complex_controls, params)
    flat_grads = strip_controls(gstate.complex_controls, grads)
    
    adam.run(None, 0, flat_controls, None, None)
    params1 = adam.update(flat_grads, flat_controls)
    params1 = slap_controls(gstate.complex_controls, params1,
                            gstate.controls_shape)
    
    assert(np.allclose(params1[0][1], params[0][1]))
    assert(np.allclose(params1[1][0], params[1][0]))


def test_sgd():
    import numpy as np

    from qoc.standard.optimizers.sgd import SGD
    
    sgd = SGD(learning_rate=1)
    params = np.ones(5)
    grads = np.ones(5)
    params = sgd.update(grads, params)
    assert(np.allclose(params, np.zeros_like(params)))


### all ###

def _test_all():
    test_constants()
    
    test_controlarea()
    test_controlnorm()
    test_controlvariation()
    test_forbiddensities()
    test_forbidstates()
    test_targetdensityinfidelity()
    test_targetdensityinfidelitytime()
    test_targetstateinfidelity()
    test_targetstateinfidelitytime()

    test_expm()
    
    test_adam()
    test_sgd()


if __name__ == "__main__":
    _test_all()

