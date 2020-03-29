"""
lqr.py
"""

from autograd.extend import Box
import autograd.numpy as anp
import numpy as np
import jax.numpy as jnp
from jax import jacrev
from jax.experimental.ode import odeint

from qoc.core.common import (strip_controls, slap_controls)
from qoc.core.mathmethods import integrate_rkdp5
from qoc.models import Dummy
from qoc.standard.optimizers import Adam
from qoc.standard.utils.autogradutil import ans_jacobian

INITIAL_TIME = 0
def lqr(costs, evolution_time, initial_astate,
        iteration_count, rhs, augmented_lagrangian=False,
        complex_controls=False,
        impose_control_conditions=None,
        initial_controls=None,
        optimizer=Adam()):
    """
    Solve an lqr problem.
    """
    pstate = Dummy()
    reporter = Dummy()
    result = Dummy()

    if augmented_lagrangian:
        for cost in costs:
            if cost.constraint is not None:
                cost.augmented_lagrangian = True
            #ENDIF
        #ENDFOR
    #ENDIF

    pstate.controls_shape = initial_controls.shape
    pstate.complex_controls = complex_controls
    pstate.costs = costs
    pstate.evolution_time = evolution_time
    pstate.initial_astate = initial_astate
    pstate.rhs = rhs
    cost = 0
    controls = initial_controls
    devolve = ans_jacobian(evolve, 0)

    for iteration in range(iteration_count):
        cost, grads = devolve(controls, reporter, pstate)
        grads = anp.conjugate(grads)
        stripped_controls = strip_controls(pstate.complex_controls,
                                           controls)
        stripped_grads = strip_controls(pstate.complex_controls,
                                        grads)
        stripped_controls = optimizer.update(grads, controls)
        controls = slap_controls(pstate.complex_controls,
                                 controls,
                                 pstate.controls_shape)
        if impose_control_conditions is not None:
            controls = impose_control_conditions(controls)
        print("{:1.8e} | {:1.8e}"
              "".format(cost, anp.linalg.norm(grads)))
    #ENDFOR
    
    result.controls = controls
    result.cost = cost
    result.final_astate = reporter.final_astate

    return result


def evolve(controls, reporter, pstate):
    """
    Evolve the state and compute the cost.
    """
    # Evolve the state.
    cost_ = 0
    final_times = anp.array([pstate.evolution_time])
    rhs_ = lambda time, astate: pstate.rhs(astate, controls, time)
    final_astate = integrate_rkdp5(
        rhs_, final_times,
        INITIAL_TIME, pstate.initial_astate,
    )
    # final_astate = odeint(
    #     rhs_, pstate.initial_astate, final_times
    # )
    reporter.final_astate = final_astate
    
    # Compute costs.
    for cost in pstate.costs:
        cost_ = cost_ + cost.cost(controls, final_astate)
    #ENDFOR

    return cost_
