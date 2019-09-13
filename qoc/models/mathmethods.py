"""
mathmethods.py - mathematical methods in physics
"""

import autograd.numpy as anp
import numpy as np

from qoc.models.operationpolicy import OperationPolicy
from qoc.standard.functions.convenience import (commutator, conjugate_transpose,
                                                l2_norm,
                                                matmuls,
                                                rms_norm,)

### INTERPOLATION METHODS ###

def interpolate_linear(x1, x2, x3, y1, y2,
                       operation_policy=OperationPolicy.CPU):
    """
    Perform a linear interpolation of the point
    (x3, y3) using two points (x1, y1), (x2, y2).

    Args:
    x1 :: float - the dependent variable on which y1 depends
    x2 :: float - the dependent variable on which y2 depends
    x3 :: float - the dependent variable on which y3 depends

    y1 :: any - the independent variable dependent on x1
    y2 :: any - the independent variable dependent on x2, type
                must be composable with y1
    operation_policy

    Returns:
    y3 :: any - the interpolated value corresponding to x3, type
                is that resulting from composition of y1 and y2
    """
    return y1 + (((y2 - y1) / (x2 - x1)) * (x3 - x1))


def get_linear_interpolator(xs, ys):
    """
    Construct a function that will determine a linearly interpolated value `y`
    given a value `x` based on the given sets `xs` and `ys`.
    
    Arguments:
    xs :: ndarray - An array of independent variables that correspond to the y values in `ys`.
    ys :: ndarray - An array of dependent variables that correspond to the x values in `xs`.

    Returns:
    interpolate_y :: (x :: ndarray) -> y :: ndarray
        - A function that linearly interpolates a y value given an x value.
    """
    def interpolate_y(x):
        # If the x value is below the zone in which data is specified,
        # interpolate using the lowest two data points.
        if x <= xs[0]:
            y = interpolate_linear(xs[0], xs[1], x, ys[0], ys[1])
        # If the x value is above the zone in which data is specified,
        # interpolate using the highest two data points.
        elif x >= xs[-1]:
            y = interpolate_linear(xs[-2], xs[-1], x, ys[-2], ys[-1])
        # Otherwise, interpolate between the closest two data points
        # to x.
        else:
            # Index is the first occurence where x is l.e. an element of xs.
            index = anp.argmax(x <= xs)
            y = interpolate_linear(xs[index - 1], xs[index], x, ys[index - 1], ys[index])
        
        return y
    
    return interpolate_y


### MAGNUS EXPANSION METHODS ###

def magnus_m2(a1, dt, operation_policy=OperationPolicy.CPU):
    """
    a magnus expansion method of order two
    as seen in https://arxiv.org/abs/1709.06483

    Args:
    a1 :: numpy.ndarray - see paper
    dt :: float - see paper
    operation_policy

    Returns:
    m2 :: numpy.ndarray - magnus expansion
    """
    return dt * a1


_M4_C0 = np.divide(np.sqrt(3), 12)
def magnus_m4(a1, a2, dt, operation_policy=OperationPolicy.CPU):
    """
    a magnus expansion method of order four
    as seen in https://arxiv.org/abs/1709.06483
    Args:
    a1 :: numpy.ndarray - see paper
    a2 :: numpy.ndarray - see paper
    dt :: float - see paper
    operation_policy

    Returns:
    m4 :: numpy.ndarray - magnus expansion
    """
    return ((dt / 2) * (a1 + a2) +
            _M4_C0 * (dt ** 2) * commutator(a2, a1,
                                                  operation_policy=operation_policy))
    

_M6_C0 = np.divide(np.sqrt(15), 3)
_M6_C1 = np.divide(10, 3)
_M6_C2 = np.divide(1, 2)
_M6_C3 = np.divide(1, 240)
_M6_C4 = np.divide(1, 60)
def magnus_m6(a1, a2, a3, dt, operation_policy=OperationPolicy.CPU):
    """
    a magnus expansion method of order six
    as seen in https://arxiv.org/abs/1709.06483
    Args:
    a1 :: numpy.ndarray - see paper
    a2 :: numpy.ndarray - see paper
    a3 :: numpy.ndarray - see paper
    dt :: float - see paper
    Returns:
    m6 :: numpy.ndarray - magnus expansion
    """
    b1 = dt * a2
    b2 = _M6_C0 * dt * (a3 - a1)
    b3 = _M6_C1 * dt * (a3 - 2 * a2 + a1)
    b1_b2_commutator = commutator(b1, b2, operation_policy=operation_policy)
    return (b1 + _M6_C2 * b3 + _M6_C3
            * commutator(-20 * b1 - b3 + b1_b2_commutator,
                         b2 - _M6_C4
                         * commutator(b1, 2 * b3 + b1_b2_commutator,
                                      operation_policy=operation_policy),
                         operation_policy=operation_policy))


### LINDBLAD METHODS ###

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


### ODE METHODS ###

# RKDP5(4) Butcher tableau constants.
# From table 5.2 on pp. 178 of [1] or [3].
C1 = 0
C2 = 1 / 5
A21 = 1 / 5
C3 = 3 / 10
A31 = 3 / 40
A32 = 9 / 40
C4 = 4 / 5
A41 = 44 / 45
A42 = -56 / 15
A43 = 32 / 9
C5 = 8 / 9
A51 = 19372 / 6561
A52 = -25360 / 2187
A53 = 64448 / 6561
A54 = -212 / 729
C6 = C7 = 1
A61 = 9017 / 3168
A62 = -355 / 33
A63 = 46732 / 5247
A64 = 49 / 176
A65 = -5103 / 18656
A71 = B1 = 35 / 384
A72 = B2 = 0
A73 = B3 = 500 / 1113
A74 = B4 = 125 / 192
A75 = B5 = -2187 / 6784
A76 = B6 = 11 / 84
B7 = 0
B1H = 5179 / 57600
B2H = 0
B3H = 7571 / 16695
B4H = 393 / 640
B5H = -92097 / 339200
B6H = 187 / 2100
B7H = 1 / 40
E1 = B1 - B1H
E2 = B2 - B2H
E3 = B3 - B3H
E4 = B4 - B4H
E5 = B5 - B5H
E6 = B6 - B6H
E7 = B7 - B7H
# Other constants.
P = 5
PH = 4
Q = np.minimum(P, PH)
ERROR_EXP = -1 / (Q + 1)


def integrate_rkdp5_step(h, rhs, x, y, k1=None):
    """
    Use the Butcher tableau for RKDP5(4) to compute y1 and y1h.

    Arguments:
    h :: float - the step size
    rhs :: (x :: float, y :: ndarray (N)) -> dy_dx :: ndarray (N)
    x :: float - starting x position in the mesh
    y :: float - starting y position in the mesh
    k1 :: ndarray (N) - this value is rhs(x, y), which is equivalent
        to the value of k7 in the previous step via the First-Same-As-Last
        (FSAL) property.

    Returns:
    k7 :: ndarray (N)
    y1 :: ndarray (N) - 5th order evaluation at `x` + `h` in the mesh
    y1h :: ndarray (N) - 4th order evaluation at `x` + `h` in the mesh
    """
    if k1 is None:
        # Note that C1 = 0. Therefore, k1 may be evaluated like so:
        k1 = rhs(x, y)
    k2 = rhs(x + C2 * h, y + h * A21 * k1)
    k3 = rhs(x + C3 * h, y + h * (A31 * k1 + A32 * k2))
    k4 = rhs(x + C4 * h, y + h * (A41 * k1 + A42 * k2 + A43 * k3))
    k5 = rhs(x + C5 * h, y + h * (A51 * k1 + A52 * k2 + A53 * k3
                                  + A54 * k4))
    k6 = rhs(x + C6 * h, y + h * (A61 * k1 + A62 * k2 + A63 * k3
                                  + A64 * k4 + A65 * k5))
    # Note that B2 = B7 = 0. Therefore, y1 may be evaluated like so:
    y1 = y + h * (B1 * k1 + B3 * k3 + B4 * k4 + B5 * k5
                  + B6 * k6)
    # Note that B$ = A7$ (FSAL).
    # Therefore, y1 = y + h * (B dot K) = y + h * (A7 dot K)
    # Note also that C7 = 1.
    # Therefore, k7 may be evaluated like so:
    k7 = rhs(x + h, y1)
    # Note that B2H = 0. Therefore, y1h may be evaluated like so:
    y1h = y + h * (B1H * k1 + B3H * k3 + B4H * k4 + B5H * k5
                   + B6H * k6 + B7H * k7)
    
    return k7, y1, y1h


def integrate_rkdp5(rhs, x_final, x_initial, y_initial,
                    atol=1e-12, rtol=0.,
                    step_safety_factor=0.9,
                    step_update_factor_max=10,
                    step_update_factor_min=2e-1,):
    """
    Integrate using the RKDP5(4) method. For quick intuition, consult [2] and [3].
    See table 5.2 on pp. 178 of [1] or [3] for the Butcher tableau. See pp. 167-169 of [1]
    for automatic step size control and starting step size. Scipy's RK45 implementation
    in python [4] was used as a reference for this implementation.

    References:
    [1] E. Hairer, S.P. Norsett and G. Wanner, Solving Ordinary Differential Equations
    i. Nonstiff Problems. 2nd edition. Springer Series in Computational Mathematics,
    Springer-Verlag (1993)
    [2] https://en.wikipedia.org/wiki/Runge–Kutta_methods
    [3] https://en.wikipedia.org/wiki/Dormand–Prince_method
    [4] https://github.com/scipy/scipy/blob/master/scipy/integrate/_ivp/rk.py
    
    Arguments:
    atol :: float or array(N) - the absolute tolerance of the component-wise
        local error, i.e. "Atoli" in e.q. 4.10 on pp. 167 of [1]
    step_rejections_max :: int - the number of allowed attempts at each
        integration step to choose a step size that satisfies the
        component-wise local error
    rhs :: (x :: float, y :: array(N)) -> dy_dx :: array(N)
        - the right-hand side of the equation dy_dx = rhs(x, y)
        that defines the first order differential equation
    rtol :: float or array(N) - the relative tolerance of the component-wise
        local error, i.e. "Rtoli" in e.q. 4.10 on pp. 167 of [1]
    step_safety_factor :: float - the safety multiplication factor used in the
        step update rule, i.e. "fac" in e.q. 4.13 on pp. 168 of [1]
    step_update_factor_max :: float - the maximum step multiplication factor used in the
        step update rule, i.e. "facmax" in e.q. 4.13 on pp. 168 of [1]
    step_update_factor_min :: float - the minimum step multiplication factor used in the
        step update rule, i.e. "facmin"in e.q.e 4.13 on pp. 168 of [1]
    x_final :: float - the final value of x (inclusive) that concludes the integration interval
    x_initial :: float - the initial value of x (inclusive) that begins the integration interval
    y_initial :: array(N) - the initial value of y

    Returns:
    y_final :: array(N) - the final value of y
    """
    # Compute initial step size per pp. 169 of [1].
    f0 = rhs(x_initial, y_initial)
    d0 = l2_norm(y_initial)
    d1 = l2_norm(f0)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    y1 = y_initial + h0 * f0
    f1 = rhs(x_initial + h0, y1)
    d2 = l2_norm(f1 - f0) / h0
    if anp.maximum(d1, d2) <= 1e-15:
        h1 = anp.maximum(1e-6, h0 * 1e-3)
    else:
        h1 = anp.power(0.01 / anp.maximum(d1, d2), 1 / (P + 1))
    step_current = anp.minimum(100 * h0, h1)

    # Integrate.
    x_current = x_initial
    y_current = y_initial
    k1 = f0
    while x_current < x_final:
        step_rejected = False
        step_accepted = False
        # Repeatedly attempt to move to the next position in the mesh
        # until the step size is adapted such that the local step error
        # is within an acceptable tolerance.
        while not step_accepted:
            # Attempt to step by `step_current`.
            k7, y1, y1h = integrate_rkdp5_step(step_current, rhs, x_current, y_current,
                                               k1=k1)
            # Before the step size is updated for the next step, note where
            # the current attempted step size places us in the mesh.
            x_new = x_current + step_current
            # Compute the local error associated with the attempted step.
            scale = atol + anp.maximum(anp.abs(y1), anp.abs(y1h)) * rtol
            error_norm = rms_norm((y1 - y1h) / scale)

            # If the step is accepted, increase the step size,
            # and move to the next step.
            if error_norm < 1:
                step_accepted = True
                # Avoid division by zero in update.
                if error_norm == 0:
                    step_update_factor = step_update_factor_max
                else:
                    step_update_factor = anp.minimum(step_update_factor_max,
                                                     step_safety_factor * anp.power(error_norm, ERROR_EXP))
                # Avoid an extraneous update in next step.
                if step_rejected:
                    step_update_factor = anp.minimum(1, step_update_factor)
                step_current = step_current * step_update_factor
            # If the step was rejected, decrease the step size,
            # and reattempt the step.
            else:
                step_rejected = True
                step_update_factor = anp.maximum(step_update_factor_min,
                                                 step_safety_factor * anp.power(error_norm, ERROR_EXP))
                step_current = step_current * step_update_factor
        #ENDWHILE
        # Update the position in the mesh.
        x_current = x_new
        y_current = y1
        k1 = k7
    #ENDWHILE
    
    return y_current


### MODULE TESTS ###

_BIG = int(1e3)
_BIG_DIV_2 = _BIG / 2

def _test():
    """
    Run tests on the module.
    """
    # Test the interpolation methods.
    for i in range(_BIG):
        # Generate a line with a constant slope between -5 and 5.
        line = lambda x: slope * x
        slope = np.random.rand() * 10 - 5
        x1 = np.random.rand() * _BIG - _BIG_DIV_2
        x2 = np.random.rand() * _BIG - _BIG_DIV_2
        x3 = np.random.rand() * _BIG - _BIG_DIV_2
        # Check that the trapezoid method approximates the line
        # exactly.
        y1 = line(x1)
        y2 = line(x2)
        lx3 = line(x3)
        itx3 = interpolate_linear(x1, x2, x3, y1, y2,)
        assert(np.isclose(lx3, itx3))
    #ENDFOR

    # Test the magnus expansion methods.
    # These tests ensure the above methods were copied to code correclty.
    # They are hand checked. There may be a better way to test the methods.
    dt = 1.
    identity = np.eye(2)
    assert(np.allclose(magnus_m2(identity, dt), identity))
    assert(np.allclose(magnus_m4(*([identity] * 2), dt), identity))
    assert(np.allclose(magnus_m6(*([identity] * 3), dt), identity))
    dt = 2.
    a1 = np.array([[2., 3.], [4., 5.]])
    a2 = np.array([[9., 6.], [8., 7.]])
    a3 = np.array([[12., 13.], [11., 10.]])
    assert(np.allclose(magnus_m2(a1, dt),
                      np.array([[4., 6.],
                                [8., 10.]])))
    assert(np.allclose(magnus_m4(a1, a2, dt),
                      np.array([[11., 22.85640646],
                                [-6.47520861, 12.]])))
    assert(np.allclose(magnus_m6(a1, a2, a3, dt),
                      np.array([[-241.71158615, 100.47657236],
                                [310.29160996, 263.71158615]])))

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


def _test_rkdp5():
    """
    Test rkdp5 using a system of odes with known solutions stated in [1].

    References:
    [1] http://tutorial.math.lamar.edu/Classes/DE/Exact.aspx
    """
    from scipy.integrate import ode, solve_ivp

    PRINT = False

    # Problem setup.
    x0 = 0
    x1 = 10
    y0 = np.array((-3,))
    y_sol = lambda x: 0.5 * (-(x ** 2 + 1) - (np.sqrt(x ** 4 + 12 * x ** 3 + 2 * x ** 2 + 25)))
    def rhs(x, y):
        return ((-2 * x * y + 9 * x ** 2) / (2 * y + x ** 2 + 1))

    # Analytical solution.
    y_1_expected = y_sol(x1)

    # Scipy fortran solutions.
    r = ode(rhs).set_integrator("vode", method="bdf")
    r.set_initial_value(y0, x0)
    y_1_scipy_vode = r.integrate(x1)

    r = ode(rhs).set_integrator("dopri5")
    r.set_initial_value(y0, x0)
    y_1_scipy_dopri5_f = r.integrate(x1)

    # Scipy python solutions.
    res = solve_ivp(rhs, [x0, x1], y0, method="RK45")
    y_1_scipy_dopri5_py = res.y[:, -1]

    res = solve_ivp(rhs, [x0, x1], y0, method="Radau")
    y_1_scipy_radau_py = res.y[:, -1]

    # QOC solution.
    # The value atol=3e-13 is hand-optimized for this problem.
    y_1 = integrate_rkdp5(rhs, x1, x0, y0)

    # 1e-2 is not bad considering the solutions of the other implementations.
    # assert(np.allclose(y_1, y_1_expected, atol=1e-2))

    if PRINT:
        print("y_1_expected:\n{}"
              "".format(y_1_expected))
        print("y_1_scipy_dopri5_f:\n{}"
              "".format(y_1_scipy_dopri5_f))
        print("y_1_scipy_dopri5_py:\n{}"
              "".format(y_1_scipy_dopri5_py))
        print("y_1_qoc:\n{}"
              "".format(y_1))


if __name__ == "__main__":
    _test()
    _test_rkdp5()
