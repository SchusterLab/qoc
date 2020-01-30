"""
mathmethods.py - mathematical methods in physics
"""

import autograd.numpy as anp
import numpy as np

from qoc.standard.functions.convenience import (commutator, conjugate_transpose,
                                                matmuls,
                                                rms_norm,)

### INTERPOLATION METHODS ###

def interpolate_linear_points(x1, x2, x3, y1, y2):
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


def interpolate_linear_set(x, xs, ys):
    """
    Interpolate a `y` value corresponding to the value `x` based on
    the corresponding values `xs` and `ys`
    
    Arguments:
    x :: float - The value to interpolate a `y` value for.
    xs :: ndarray (N) - An array of independent variables that correspond to the y values in `ys`.
        It is assumed that `xs` is sorted.
    ys :: ndarray (N x y_shape)- An array of dependent variables that correspond to the x values in `xs`.
        It is assumed that `ys` is sorted such that each index corresponds to the y value that
        matches the corresponding x value at the same index in `xs`.

    Returns:
    y :: ndarray (y_shape) - the `y` value that corresponds to `x`.
    """
    # If the x value is below the zone in which data is specified,
    # interpolate using the lowest two data points.
    if x <= xs[0]:
        y = interpolate_linear_points(xs[0], xs[1], x, ys[0], ys[1])
    # If the x value is above the zone in which data is specified,
    # interpolate using the highest two data points.
    elif x >= xs[-1]:
        y = interpolate_linear_points(xs[-2], xs[-1], x, ys[-2], ys[-1])
    # Otherwise, interpolate between the closest two data points
    # to x.
    else:
        # Index is the first occurence where x is l.e. an element of xs.
        index = anp.argmax(x <= xs)
        y = interpolate_linear_points(xs[index - 1], xs[index], x, ys[index - 1], ys[index])
        
    return y


### MAGNUS EXPANSION METHODS ###

_M2_C1 = 0.5

def magnus_m2(a, dt, time):
    """
    Construct a magnus expasion of `a` of order two.
    
    References:
    [1] https://arxiv.org/abs/1709.06483

    Arguments:
    a :: (time :: float) -> ndarray (a_shape)
        - the matrix to expand
    dt :: float - the time step
    time :: float - the current time

    Returns:
    m2 :: ndarray (a_shape) - magnus expansion
    """
    t1 = time + dt * _M2_C1
    a1 = a(t1)
    m2 = dt * a1
    return m2


_M4_C1 = 0.5 - np.divide(np.sqrt(3), 6)
_M4_C2 = 0.5 + np.divide(np.sqrt(3), 6)
_M4_F0 = np.divide(np.sqrt(3), 12)

def magnus_m4(a, dt, time):
    """
    Construct a magnus expasion of `a` of order four.
    
    References:
    [1] https://arxiv.org/abs/1709.06483

    Arguments:
    a :: (time :: float) -> ndarray (a_shape)
        - the matrix to expand
    dt :: float - the time step
    time :: float - the current time

    Returns:
    m4 :: ndarray (a_shape) - magnus expansion
    """
    t1 = time + dt * _M4_C1
    t2 = time + dt * _M4_C2
    a1 = a(t1)
    a2 = a(t2)
    m4 = ((dt / 2) * (a1 + a2)
          + _M4_F0 * (dt ** 2) * commutator(a2, a1))
    return m4


_M6_C1 = 0.5 - np.divide(np.sqrt(15), 10)
_M6_C2 = 0.5
_M6_C3 = 0.5 + np.divide(np.sqrt(15), 10)
_M6_F0 = np.divide(np.sqrt(15), 3)
_M6_F1 = np.divide(10, 3)
_M6_F2 = np.divide(1, 2)
_M6_F3 = np.divide(1, 240)
_M6_F4 = np.divide(1, 60)

def magnus_m6(a, dt, time):
    """
    Construct a magnus expasion of `a` of order six.
    
    References:
    [1] https://arxiv.org/abs/1709.06483

    Arguments:
    a :: (time :: float) -> ndarray (a_shape)
        - the matrix to expand
    dt :: float - the time step
    time :: float - the current time

    Returns:
    m6 :: ndarray (a_shape) - magnus expansion
    """
    t1 = time + dt * _M6_C1
    t2 = time + dt * _M6_C2
    t3 = time + dt * _M6_C3
    a1 = a(t1)
    a2 = a(t2)
    a3 = a(t3)
    b1 = dt * a2
    b2 = _M6_F0 * dt * (a3 - a1)
    b3 = _M6_F1 * dt * (a3 - 2 * a2 + a1)
    b1_b2_commutator = commutator(b1, b2)
    m6 = (b1 + _M6_F2 * b3 + _M6_F3
            * commutator(-20 * b1 - b3 + b1_b2_commutator,
                         b2 - _M6_F4
                         * commutator(b1, 2 * b3 + b1_b2_commutator)))
    return m6


### LINDBLAD METHODS ###

def get_lindbladian(densities, dissipators=None, hamiltonian=None,
                    operators=None,):
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
    if hamiltonian is not None:
        lindbladian = -1j * commutator(hamiltonian, densities,)
    else:
        lindbladian = 0
        
    if dissipators is not None and operators is not None:
        operators_dagger = conjugate_transpose(operators,)
        operators_product = matmuls(operators_dagger, operators,)
        for i, operator in enumerate(operators):
            dissipator = dissipators[i]
            operator_dagger = operators_dagger[i]
            operator_product = operators_product[i]
            lindbladian = (lindbladian
                           + (dissipator
                              * (matmuls(operator, densities, operator_dagger,)
                                 - 0.5 * matmuls(operator_product, densities,)
                                 - 0.5 * matmuls(densities, operator_product,))))
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
# RKDP5(4) dense output constants from [5].
D1 = -12715105075 / 11282082432
D2 = 0
D3 = 87487479700 / 32700410799
D4 = -10690763975 / 1880347072
D5 = 701980252875 / 199316789632
D6 = -1453857185 / 822651844
D7 = 69997945 / 29380423
# RKDP5(4) method constants.
P = 5
PH = 4
Q = np.minimum(P, PH)
ERROR_EXP = -1 / (Q + 1)


def rkdp5_dense(ks, x0, x1, x_eval_step, y0, y1):
    """
    Interpolate values between a step using a quartic polynomial.
    See [5] for the disambiguation of where this method comes from.
    
    Arguments:
    ks :: ndarray (7) - the k values for this step
    x0 :: float - the initial x value for this step
    x1 :: float - the final x value for this step
    x_eval_step :: ndarray (eval_step_count) - This is an array of 
        x values whose corresponding y value should
        be obtained. It is assumed that the values in this
        array lie between x0 and x1 inclusive,
        that this array does not contain duplicates,
        and that the values are sorted in increasing order.
    y0 :: ndarray (N) - the y value corresponding to `x0`
    y1 :: ndarray (N) - the y value corresponding to `x1`

    Returns:
    y_eval_step :: ndarray (eval_step_count) - The y values corresponding
        to the x values in `x_eval_step`.
    """
    # Interpolate.
    h = x1 - x0
    r1 = y0
    r2 = y1 - y0
    r3 = y0 + h * ks[0] - y1
    r4 = 2 * (y1 - y0) - h * (ks[0] + ks[6])
    # Note that D2=0. Therefore, we may compute r5 like so:
    r5 = h * (D1 * ks[0] + D3 * ks[2] + D4 * ks[3] + D5 * ks[4]
              + D6 * ks[5] + D7 * ks[6])
    theta = (x_eval_step - x0) / h
    theta2 = theta ** 2
    theta3 = theta ** 3
    theta4 = theta2 ** 2
    y_eval_step = (r1
                   + theta * (r2 + r3)
                   - theta2 * (r3 - r4 -r5)
                   - theta3 * (r4 + 2 * r5)
                   + theta4 * r5)

    return y_eval_step


def integrate_rkdp5_step(h, rhs, x0, y0, k1=None):
    """
    Use the Butcher tableau for RKDP5(4) to compute y1 and y1h.

    Arguments:
    h :: float - the step size
    rhs :: (x :: float, y :: ndarray (N)) -> dy_dx :: ndarray (N)
    x0 :: float - starting x position in the mesh
    y0 :: ndarray (N) - starting y position in the mesh
    k1 :: ndarray (N) - this value is rhs(x0, y0), which is equivalent
        to the value of k7 in the previous step via the First-Same-As-Last
        (FSAL) property.

    Returns:
    ks :: ndarray (N x 7) - the k values for this step
    y1 :: ndarray (N) - 5th order evaluation at `x0` + `h` in the mesh
    y1h :: ndarray (N) - 4th order evaluation at `x0` + `h` in the mesh
    """
    if k1 is None:
        # Note that C1 = 0. Therefore, k1 may be evaluated like so:
        k1 = rhs(x0, y0)
    k2 = rhs(x0 + C2 * h, y0 + h * A21 * k1)
    k3 = rhs(x0 + C3 * h, y0 + h * (A31 * k1 + A32 * k2))
    k4 = rhs(x0 + C4 * h, y0 + h * (A41 * k1 + A42 * k2 + A43 * k3))
    k5 = rhs(x0 + C5 * h, y0 + h * (A51 * k1 + A52 * k2 + A53 * k3
                                    + A54 * k4))
    k6 = rhs(x0 + C6 * h, y0 + h * (A61 * k1 + A62 * k2 + A63 * k3
                                    + A64 * k4 + A65 * k5))
    # Note that B2 = B7 = 0. Therefore, y1 may be evaluated like so:
    y1 = y0 + h * (B1 * k1 + B3 * k3 + B4 * k4 + B5 * k5
                   + B6 * k6)
    # Note that B$ = A7$ (FSAL).
    # Therefore, y1 = y0 + h * (B dot K) = y0 + h * (A7 dot K)
    # Note also that C7 = 1.
    # Therefore, k7 may be evaluated like so:
    k7 = rhs(x0 + h, y1)
    # Note that B2H = 0. Therefore, y1h may be evaluated like so:
    y1h = y0 + h * (B1H * k1 + B3H * k3 + B4H * k4 + B5H * k5
                    + B6H * k6 + B7H * k7)
    
    ks = (k1, k2, k3, k4, k5, k6, k7)

    return ks, y1, y1h


def integrate_rkdp5(rhs, x_eval, x_initial, y_initial,
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
    [5] https://math.stackexchange.com/questions/2947231/how-can-i-derive-the-dense-output-of-ode45/2948244
    
    Arguments:
    atol :: float or array(N) - the absolute tolerance of the component-wise
        local error, i.e. "Atoli" in e.q. 4.10 on pp. 167 of [1]
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
    x_eval :: ndarray (eval_count) - an array of points `x` whose
        corresponding `y` value should be evaluated. It is assumed
        that this list does not contain duplicates, that
        the values are sorted in increasing order, and that
        all values are greater than `x_initial`.
    x_final :: float - the final value of x (inclusive) that concludes the integration interval
    x_initial :: float - the initial value of x (inclusive) that begins the integration interval
    y_initial :: array(N) - the initial value of y

    Returns:
    y_evald :: ndarray (eval_count x N) - an array of points `y` whose
        corresponding `x` value is specified in x_eval
    """
    # Determine how far to integrate to.
    if len(x_eval) == 0:
        raise ValueError("No output was specified.")
    else:
        x_final = x_eval[-1]
    
    # Compute initial step size per pp. 169 of [1].
    f0 = rhs(x_initial, y_initial)
    d0 = rms_norm(y_initial)
    d1 = rms_norm(f0)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    y1 = y_initial + h0 * f0
    f1 = rhs(x_initial + h0, y1)
    d2 = rms_norm(f1 - f0) / h0
    if anp.maximum(d1, d2) <= 1e-15:
        h1 = anp.maximum(1e-6, h0 * 1e-3)
    else:
        h1 = anp.power(0.01 / anp.maximum(d1, d2), 1 / (P + 1))
    step_current = anp.minimum(100 * h0, h1)

    # Integrate.
    y_eval_list = list()
    x_current = x_initial
    y_current = y_initial
    k1 = f0
    while x_current <= x_final:
        step_rejected = False
        step_accepted = False
        # Repeatedly attempt to move to the next position in the mesh
        # until the step size is adapted such that the local step error
        # is within an acceptable tolerance.
        while not step_accepted:
            # Attempt to step by `step_current`.
            ks, y1, y1h = integrate_rkdp5_step(step_current, rhs, x_current, y_current,
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
        # Interpolate any output points that ocurred in the step.
        x_eval_step_indices = anp.nonzero(anp.logical_and(x_current <= x_eval, x_eval <= x_new))[0]
        x_eval_step = x_eval[x_eval_step_indices]
        if len(x_eval_step) != 0:
            y_eval_step = rkdp5_dense(ks, x_current, x_new, x_eval_step, y_current, y1)
            for y_eval_ in y_eval_step:
                y_eval_list.append(y_eval_)
        
        # Update the position in the mesh.
        x_current = x_new
        y_current = y1
        k1 = ks[6] # k[6] = k7
    #ENDWHILE
    
    return anp.stack(y_eval_list)
