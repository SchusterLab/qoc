"""
gld.py - a module to evolve a set of density matrices
under the lindblad master equation using discrete
control parameters
"""

import autograd.numpy as anp

def grape_lindblad_discrete():
    pass


def _evaluate_lindblad(controls, pstate, reporter):
    """
    Evolve a set of density matrices under the lindblad equation
    and compute associated optimization costs.

    Args:
    controls :: ndarray - the control parameters
    pstate :: qoc.models.GrapeLindbladDiscreteState
        or qoc.models.EvolveLindbladDiscreteState - the program state
    reporter :: any - the object to keep track of relevant information

    Returns:
    total_error :: float - the optimization cost for the provided controls
    """
    # Initialize local variables (heap -> stack).
    costs = pstate.costs
    densities = pstate.initial_densities
    dt = pstate.dt
    evolve_lindblad = pstate.evolve_lindblad
    final_control_step = pstate.final_control_step
    final_system_step = pstate.final_system_step
    step_costs = pstate.step_costs
    system_step_multiplier = pstate.system_step_multiplier
    total_error = 0

    for system_step in range(final_system_step + 1):
        control_step, _ = anp.divmod(system_step, system_step_multiplier)
        is_final_control_step = control_step == final_control_step
        is_final_system_step = system_step == final_system_step
        time = system_step * dt

        # Evolve the density matrices.
        densities = evolve_lindblad(controls, control_step, densities, dt,
                                    time, is_final_control_step)

        # Compute the costs.
        if is_final_system_step:
            for i, cost in enumerate(costs):
                error = cost.cost(controls, densities, system_step)
                total_error = total_error + error
            #ENDFOR
            reporter.last_densities = densities
            reporter.last_total_error = total_error
        else:
            for i, step_cost in enumerate(step_costs):
                error = step_cost.cost(controls, densities, system_step)
                total_error = total_error + error
            #ENDFOR
    #ENDFOR

    return total_error


def _test():
    pass

if __name__ == "__main__":
    _test()
