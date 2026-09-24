"""Compiled (Numba) trajectory integrator: a faithful re-implementation of the scipy path.

Place in the program
--------------------
    solve.run_trajectory(..., method="numba")
        -> solve.initial_state(..., quadrature="gauss")  -> fast.incoming_leg_moments
        -> fast.integrate_scattering                     (this module, compiled)
        -> the same outcome classification as method="reference"
The scipy path (method="reference", solve.integrate_state -> solve_ivp) stays the oracle.

Equations: identical to solve.equations_of_motion (see model.py / solve.py docstrings); the
right-hand side here is the same formulas written out component by component.

Numerical method, and why it is the SAME method
-----------------------------------------------
Explicit Runge-Kutta DOP853 (Hairer-Norsett-Wanner), coded to follow scipy 1.9's
scipy.integrate._ivp.rk.DOP853 and solve_ivp step for step:
  * Butcher tableau, error weights E3/E5 and dense-output matrix D imported from
    scipy.integrate._ivp.dop853_coefficients (not retyped);
  * error norm  |h| ||err5||^2 / sqrt((||err5||^2 + 0.01 ||err3||^2) n)  with
    scale = atol + max(|y|, |y_new|) rtol;
  * step control: SAFETY 0.9, factor bounds [0.2, 10], exponent -1/8, no growth right
    after a rejection, min_step = 10 ulp(t);
  * first step from scipy's select_initial_step (Hairer II.4);
  * 7th-order dense output (3 extra stages) for event location;
  * events: sign change test between accepted steps with the direction convention of
    scipy.integrate._ivp.ivp.find_active_events, roots by Brent's method with
    xtol = rtol = 4 eps (a transcription of scipy's brentq.c), terminal handling and event
    ordering as in handle_events.
So for the same inputs the two paths take essentially the same steps; differences come
only from floating-point evaluation order and are amplified only where the dynamics is
sensitive (near capture edges). tests/test_fast.py measures this.

Events (index: meaning, direction, terminal)
  0 contact   r - 2                       -1   always
  1 periapsis r . v                       +1   never
  2 escape    r'|r'|/2 - 2/r^3            +1   if stop == "escape"
  3 r_out     r - r_out                   +1   if stop == "r_out"

Also compiled: validation diagnostics at every accepted step (E, J, max_i ||m_i| - 1|), so
the Python wrapper can compute the same drift numbers as the reference path without a
Python loop over steps; and the incoming-leg quadrature (Gauss-Legendre, see
incoming_leg_moments) that replaces quad_vec in the in-state correction.

Interfaces
----------
integrate_scattering(y0, t_bound, inertia, rtol, atol, outgoing_radius, escape_terminal,
                     rout_terminal) -> tuple (see its docstring)
incoming_leg_moments(x_start, b, m1, m2) -> (12,) torque moments (Int tau1, Int tau2,
                     Int s tau1, Int s tau2); solve.incoming_leg_correction divides them by
                     I v and I v^2
Compilation happens on first call (cache=True writes it to __pycache__).
"""

import math

import numpy as np
try:
    from numba import njit
except ImportError as exc:     # numba is a declared dependency; say so clearly if it is missing
    raise ImportError("method='numba' needs the numba package (pip install numba); "
                      "method='reference' (scipy) works without it") from exc
from scipy.integrate._ivp import dop853_coefficients as dop

# --- DOP853 constants (scipy's own arrays) -------------------------------------------------
N_STAGES = dop.N_STAGES                      # 12
N_EXTENDED = dop.N_STAGES_EXTENDED           # 16
A_FULL = np.ascontiguousarray(dop.A)         # (16, 16)
B_MAIN = np.ascontiguousarray(dop.B)         # (12,)
C_FULL = np.ascontiguousarray(dop.C)         # (16,)
E3 = np.ascontiguousarray(dop.E3)            # (13,)
E5 = np.ascontiguousarray(dop.E5)            # (13,)
D_DENSE = np.ascontiguousarray(dop.D)        # (4, 16)
SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10.0
ERROR_EXPONENT = -1.0 / 8.0                  # -1 / (error_estimator_order + 1)
EPS = np.finfo(float).eps
STATE = 19
CONTACT_SEPARATION = 2.0
ESCAPE_COEFF = 2.0                           # = -MIN_RADIAL_FORCE_COEFF / 3 (model.py)


# --- physics ---------------------------------------------------------------------------------

@njit(cache=True)
def rhs(y, out, inertia):
    """dy/dt, component-wise transcription of solve.equations_of_motion."""
    x, yy, z = y[0], y[1], y[2]
    r = np.sqrt(x * x + yy * yy + z * z)
    hx, hy, hz = x / r, yy / r, z / r
    m1x, m1y, m1z = y[6], y[7], y[8]
    m2x, m2y, m2z = y[9], y[10], y[11]
    w1x, w1y, w1z = y[12], y[13], y[14]
    w2x, w2y, w2z = y[15], y[16], y[17]
    a1 = m1x * hx + m1y * hy + m1z * hz
    a2 = m2x * hx + m2y * hy + m2z * hz
    m12 = m1x * m2x + m1y * m2y + m1z * m2z
    # force on sphere 2 = relative acceleration
    f = 3.0 / r ** 4
    c = m12 - 5.0 * a1 * a2
    out[0], out[1], out[2] = y[3], y[4], y[5]
    out[3] = f * (a1 * m2x + a2 * m1x + c * hx)
    out[4] = f * (a1 * m2y + a2 * m1y + c * hy)
    out[5] = f * (a1 * m2z + a2 * m1z + c * hz)
    # moments: m' = w x m
    out[6] = w1y * m1z - w1z * m1y
    out[7] = w1z * m1x - w1x * m1z
    out[8] = w1x * m1y - w1y * m1x
    out[9] = w2y * m2z - w2z * m2y
    out[10] = w2z * m2x - w2x * m2z
    out[11] = w2x * m2y - w2y * m2x
    # torques: tau1 = (3 a2 m1 x h - m1 x m2)/r^3, tau2 = (3 a1 m2 x h + m1 x m2)/r^3
    g = 1.0 / (r ** 3 * inertia)
    cx = m1y * m2z - m1z * m2y
    cy = m1z * m2x - m1x * m2z
    cz = m1x * m2y - m1y * m2x
    out[12] = g * (3.0 * a2 * (m1y * hz - m1z * hy) - cx)
    out[13] = g * (3.0 * a2 * (m1z * hx - m1x * hz) - cy)
    out[14] = g * (3.0 * a2 * (m1x * hy - m1y * hx) - cz)
    out[15] = g * (3.0 * a1 * (m2y * hz - m2z * hy) + cx)
    out[16] = g * (3.0 * a1 * (m2z * hx - m2x * hz) + cy)
    out[17] = g * (3.0 * a1 * (m2x * hy - m2y * hx) + cz)
    rho_sq = x * x + yy * yy
    out[18] = (x * y[4] - yy * y[3]) / rho_sq if rho_sq > 0 else 0.0


@njit(cache=True)
def torques_on_line(x, b, m1, m2, out):
    """(tau1, tau2) at position (x, b, 0) into out[0:6] (for the incoming-leg quadrature)."""
    r = np.sqrt(x * x + b * b)
    hx, hy = x / r, b / r
    a1 = m1[0] * hx + m1[1] * hy
    a2 = m2[0] * hx + m2[1] * hy
    cx = m1[1] * m2[2] - m1[2] * m2[1]
    cy = m1[2] * m2[0] - m1[0] * m2[2]
    cz = m1[0] * m2[1] - m1[1] * m2[0]
    g = 1.0 / r ** 3
    out[0] = g * (3.0 * a2 * (-m1[2] * hy) - cx)
    out[1] = g * (3.0 * a2 * (m1[2] * hx) - cy)
    out[2] = g * (3.0 * a2 * (m1[0] * hy - m1[1] * hx) - cz)
    out[3] = g * (3.0 * a1 * (-m2[2] * hy) + cx)
    out[4] = g * (3.0 * a1 * (m2[2] * hx) + cy)
    out[5] = g * (3.0 * a1 * (m2[0] * hy - m2[1] * hx) + cz)


@njit(cache=True)
def diagnostics(y, inertia, out):
    """out = [E, Jx, Jy, Jz, max_i | |m_i| - 1 |] for one state (observables.py definitions)."""
    x, yy, z = y[0], y[1], y[2]
    r = np.sqrt(x * x + yy * yy + z * z)
    hx, hy, hz = x / r, yy / r, z / r
    a1 = y[6] * hx + y[7] * hy + y[8] * hz
    a2 = y[9] * hx + y[10] * hy + y[11] * hz
    m12 = y[6] * y[9] + y[7] * y[10] + y[8] * y[11]
    kinetic = 0.5 * (y[3] ** 2 + y[4] ** 2 + y[5] ** 2)
    rotational = 0.5 * inertia * (y[12] ** 2 + y[13] ** 2 + y[14] ** 2 + y[15] ** 2 + y[16] ** 2 + y[17] ** 2)
    out[0] = kinetic + rotational - (3.0 * a1 * a2 - m12) / r ** 3
    out[1] = yy * y[5] - z * y[4] + inertia * (y[12] + y[15])
    out[2] = z * y[3] - x * y[5] + inertia * (y[13] + y[16])
    out[3] = x * y[4] - yy * y[3] + inertia * (y[14] + y[17])
    n1 = abs(np.sqrt(y[6] ** 2 + y[7] ** 2 + y[8] ** 2) - 1.0)
    n2 = abs(np.sqrt(y[9] ** 2 + y[10] ** 2 + y[11] ** 2) - 1.0)
    out[4] = max(n1, n2)


@njit(cache=True)
def event_values(y, outgoing_radius, out):
    r = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
    rdotv = y[0] * y[3] + y[1] * y[4] + y[2] * y[5]
    radial_velocity = rdotv / r
    out[0] = r - CONTACT_SEPARATION
    out[1] = rdotv
    out[2] = 0.5 * radial_velocity * abs(radial_velocity) - ESCAPE_COEFF / r ** 3
    out[3] = r - outgoing_radius


# --- DOP853 pieces ---------------------------------------------------------------------------

@njit(cache=True)
def rms_norm(v):
    s = 0.0
    for k in range(v.size):
        s += v[k] * v[k]
    return np.sqrt(s) / np.sqrt(v.size)


@njit(cache=True)
def select_initial_step(y0, f0, inertia, rtol, atol, work, fwork):
    """scipy.integrate._ivp.common.select_initial_step for order 7, direction +1."""
    n = y0.size
    for k in range(n):
        work[k] = atol + abs(y0[k]) * rtol                    # scale
    d0 = 0.0
    d1 = 0.0
    for k in range(n):
        d0 += (y0[k] / work[k]) ** 2
        d1 += (f0[k] / work[k]) ** 2
    d0 = np.sqrt(d0) / np.sqrt(n)
    d1 = np.sqrt(d1) / np.sqrt(n)
    h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
    y1 = y0 + h0 * f0
    rhs(y1, fwork, inertia)
    d2 = 0.0
    for k in range(n):
        d2 += ((fwork[k] - f0[k]) / work[k]) ** 2
    d2 = np.sqrt(d2) / np.sqrt(n) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1.0 / 8.0)
    return min(100.0 * h0, h1)


@njit(cache=True)
def dense_coefficients(t_old, h, y_old, y_new, f_new, K, inertia, F, tmp):
    """3 extra stages and the 7 x n interpolation matrix F (scipy DOP853._dense_output_impl)."""
    n = y_old.size
    for s in range(N_STAGES + 1, N_EXTENDED):
        for k in range(n):
            acc = 0.0
            for j in range(s):
                acc += A_FULL[s, j] * K[j, k]
            tmp[k] = y_old[k] + h * acc
        rhs(tmp, K[s], inertia)
    for k in range(n):
        delta = y_new[k] - y_old[k]
        F[0, k] = delta
        F[1, k] = h * K[0, k] - delta
        F[2, k] = 2.0 * delta - h * (f_new[k] + K[0, k])
        for row in range(4):
            acc = 0.0
            for j in range(N_EXTENDED):
                acc += D_DENSE[row, j] * K[j, k]
            F[3 + row, k] = h * acc


@njit(cache=True)
def dense_eval(t, t_old, h, y_old, F, out):
    """Dop853DenseOutput._call_impl for a scalar t."""
    x = (t - t_old) / h
    n = y_old.size
    for k in range(n):
        acc = 0.0
        for i in range(7):
            acc += F[6 - i, k]
            if i % 2 == 0:
                acc *= x
            else:
                acc *= 1.0 - x
        out[k] = acc + y_old[k]


@njit(cache=True)
def event_at(t, index, t_old, h, y_old, F, outgoing_radius, ytmp, gtmp):
    dense_eval(t, t_old, h, y_old, F, ytmp)
    event_values(ytmp, outgoing_radius, gtmp)
    return gtmp[index]


@njit(cache=True)
def brent_root(index, xa, xb, t_old, h, y_old, F, outgoing_radius, ytmp, gtmp):
    """Transcription of scipy/optimize/Zeros/brentq.c with xtol = rtol = 4 eps, iter = 100."""
    xtol = 4.0 * EPS
    rtol = 4.0 * EPS
    xpre, xcur = xa, xb
    xblk, fblk, spre, scur = 0.0, 0.0, 0.0, 0.0
    fpre = event_at(xpre, index, t_old, h, y_old, F, outgoing_radius, ytmp, gtmp)
    fcur = event_at(xcur, index, t_old, h, y_old, F, outgoing_radius, ytmp, gtmp)
    if fpre == 0.0:
        return xpre
    if fcur == 0.0:
        return xcur
    for _ in range(100):
        if fpre * fcur < 0.0:
            xblk, fblk = xpre, fpre
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre, xcur, xblk = xcur, xblk, xcur
            fpre, fcur, fblk = fcur, fblk, fcur
        delta = (xtol + rtol * abs(xcur)) / 2.0
        sbis = (xblk - xcur) / 2.0
        if fcur == 0.0 or abs(sbis) < delta:
            return xcur
        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            if 2.0 * abs(stry) < min(abs(spre), 3.0 * abs(sbis) - delta):
                spre, scur = scur, stry
            else:
                spre, scur = sbis, sbis
        else:
            spre, scur = sbis, sbis
        xpre, fpre = xcur, fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta
        fcur = event_at(xcur, index, t_old, h, y_old, F, outgoing_radius, ytmp, gtmp)
    return xcur


@njit(cache=True)
def ulp(t):
    """np.nextafter(t, inf) - t for t >= 0 (scipy's min_step uses 10 of these)."""
    if t == 0.0:
        return 5e-324
    mantissa, exponent = math.frexp(abs(t))
    return 2.0 ** (exponent - 53)


@njit(cache=True)
def integrate_scattering(y0, t_bound, inertia, rtol, atol, outgoing_radius,
                         escape_terminal, rout_terminal):
    """Integrate from t = 0 to t_bound with the 4 scattering events.

    Returns (status, n_rhs, t_final, y_final, event_counts (4,), event_times (4, cap),
             event_states (4, cap, 19), step_times (n_steps,), step_diagnostics (n_steps, 5))
    status: 0 reached t_bound, 1 terminal event, -1 step size underflow (solver failure).
    step_* include the initial state and, after a terminal event, the state at the event
    (exactly the columns solve_ivp would return in sol.t / sol.y).
    """
    n = y0.size
    direction = np.array([-1.0, 1.0, 1.0, 1.0])
    terminal = np.array([True, False, escape_terminal, rout_terminal])

    K = np.empty((N_EXTENDED, n))
    F = np.empty((7, n))
    y = y0.copy()
    y_new = np.empty(n)
    f = np.empty(n)
    f_new = np.empty(n)
    tmp = np.empty(n)
    scale = np.empty(n)
    ytmp = np.empty(n)
    g = np.empty(4)
    g_new = np.empty(4)
    gtmp = np.empty(4)
    diag = np.empty(5)
    y_final = np.empty(n)

    cap = 64
    ev_count = np.zeros(4, dtype=np.int64)
    ev_t = np.empty((4, cap))
    ev_y = np.empty((4, cap, n))
    step_cap = 4096
    step_t = np.empty(step_cap)
    step_d = np.empty((step_cap, 5))
    n_steps = 0

    t = 0.0
    rhs(y, f, inertia)
    n_rhs = 1
    h_abs = select_initial_step(y, f, inertia, rtol, atol, scale, tmp)
    n_rhs += 1
    event_values(y, outgoing_radius, g)
    diagnostics(y, inertia, diag)
    step_t[0] = t
    step_d[0] = diag
    n_steps = 1
    status = 2   # running

    while status == 2:
        # ---- one accepted step (RungeKutta._step_impl) --------------------------------
        min_step = 10.0 * ulp(t)
        if h_abs < min_step:
            h_abs = min_step
        step_rejected = False
        while True:
            if h_abs < min_step:
                return -1, n_rhs, t, y, ev_count, ev_t, ev_y, step_t[:n_steps], step_d[:n_steps]
            h = h_abs
            t_new = t + h
            if t_new - t_bound > 0:
                t_new = t_bound
            h = t_new - t
            h_abs = abs(h)
            # stages
            for k in range(n):
                K[0, k] = f[k]
            for s in range(1, N_STAGES):
                for k in range(n):
                    acc = 0.0
                    for j in range(s):
                        acc += A_FULL[s, j] * K[j, k]
                    tmp[k] = y[k] + h * acc
                rhs(tmp, K[s], inertia)
            for k in range(n):
                acc = 0.0
                for j in range(N_STAGES):
                    acc += B_MAIN[j] * K[j, k]
                y_new[k] = y[k] + h * acc
            rhs(y_new, f_new, inertia)
            n_rhs += N_STAGES
            for k in range(n):
                K[N_STAGES, k] = f_new[k]
            # error norm: DOP853._estimate_error_norm, which aggregates GLOBALLY,
            #   |h| ||err5||^2 / sqrt((||err5||^2 + 0.01 ||err3||^2) n),
            # (the per-component hypot correction is DOP853._estimate_error, "left for testing
            # purposes" and not used by step control in scipy 1.9). Identical RHS-evaluation
            # counts vs scipy in tests/test_fast.py confirm the same accept/reject sequence.
            e5 = 0.0
            e3 = 0.0
            for k in range(n):
                sc = atol + max(abs(y[k]), abs(y_new[k])) * rtol
                a5 = 0.0
                a3 = 0.0
                for j in range(N_STAGES + 1):
                    a5 += K[j, k] * E5[j]
                    a3 += K[j, k] * E3[j]
                e5 += (a5 / sc) ** 2
                e3 += (a3 / sc) ** 2
            if e5 == 0.0 and e3 == 0.0:
                error_norm = 0.0
            else:
                error_norm = abs(h) * e5 / np.sqrt((e5 + 0.01 * e3) * n)
            if error_norm < 1.0:
                if error_norm == 0.0:
                    factor = MAX_FACTOR
                else:
                    factor = min(MAX_FACTOR, SAFETY * error_norm ** ERROR_EXPONENT)
                if step_rejected:
                    factor = min(1.0, factor)
                h_abs *= factor
                break
            h_abs *= max(MIN_FACTOR, SAFETY * error_norm ** ERROR_EXPONENT)
            step_rejected = True

        t_old = t
        h_step = t_new - t_old
        # ---- events (solve_ivp loop) ------------------------------------------------------
        event_values(y_new, outgoing_radius, g_new)
        t_record = t_new
        dense_ready = False
        roots = np.full(4, np.inf)
        active = np.zeros(4, dtype=np.bool_)
        any_active = False
        for e in range(4):
            up = g[e] <= 0.0 and g_new[e] >= 0.0
            down = g[e] >= 0.0 and g_new[e] <= 0.0
            if (up and direction[e] > 0) or (down and direction[e] < 0):
                active[e] = True
                any_active = True
        if any_active:
            dense_coefficients(t_old, h_step, y, y_new, f_new, K, inertia, F, tmp)
            n_rhs += 3
            dense_ready = True
            for e in range(4):
                if active[e]:
                    roots[e] = brent_root(e, t_old, t_new, t_old, h_step, y, F, outgoing_radius, ytmp, gtmp)
            # handle_events: if any active event is terminal, keep events up to the first
            # terminal one in time order.
            cutoff = np.inf
            for e in range(4):
                if active[e] and terminal[e] and roots[e] < cutoff:
                    cutoff = roots[e]
            order = np.argsort(roots)
            for idx in range(4):
                e = order[idx]
                if not active[e] or roots[e] > cutoff:
                    continue
                if ev_count[e] == ev_t.shape[1]:          # grow event storage x2
                    new_t = np.empty((4, 2 * ev_t.shape[1]))
                    new_t[:, :ev_t.shape[1]] = ev_t
                    new_y = np.empty((4, 2 * ev_y.shape[1], n))
                    new_y[:, :ev_y.shape[1]] = ev_y
                    ev_t, ev_y = new_t, new_y
                ev_t[e, ev_count[e]] = roots[e]
                dense_eval(roots[e], t_old, h_step, y, F, ev_y[e, ev_count[e]])
                ev_count[e] += 1
            if cutoff < np.inf:
                status = 1
                t_record = cutoff
        for e in range(4):
            g[e] = g_new[e]

        # ---- bookkeeping --------------------------------------------------------------------
        if status == 1:
            dense_eval(t_record, t_old, h_step, y, F, ytmp)
            diagnostics(ytmp, inertia, diag)
            y_final = ytmp.copy()
        else:
            diagnostics(y_new, inertia, diag)
        if n_steps == step_t.size:                     # grow step storage x2
            new_st = np.empty(2 * step_t.size)
            new_st[:n_steps] = step_t
            new_sd = np.empty((2 * step_t.size, 5))
            new_sd[:n_steps] = step_d
            step_t, step_d = new_st, new_sd
        step_t[n_steps] = t_record
        step_d[n_steps] = diag
        n_steps += 1

        for k in range(n):
            y[k] = y_new[k]
            f[k] = f_new[k]
        t = t_new
        if status == 2 and t - t_bound >= 0:
            status = 0
    if status == 1:
        return status, n_rhs, t_record, y_final, ev_count, ev_t, ev_y, step_t[:n_steps], step_d[:n_steps]
    return status, n_rhs, t, y, ev_count, ev_t, ev_y, step_t[:n_steps], step_d[:n_steps]


# --- in-state quadrature --------------------------------------------------------------------
# Int_0^inf over s with the map s = L u / (1 - u), u in [0, 1): ds = L / (1 - u)^2 du.
# The torque integrand decays like s^-3 and s * torque like s^-2, so both mapped integrands
# are smooth and bounded at u -> 1. L = |x_start| (the scale on which the integrand varies).
# Composite Gauss-Legendre: N_PANELS panels of N_NODES nodes, geometrically graded toward
# u = 0 where the integrand varies fastest. Accuracy vs quad_vec measured in tests/test_fast.py.
GL_NODES, GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
PANEL_EDGES = np.array([0.0, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0])


@njit(cache=True)
def incoming_leg_moments(x_start, b, m1, m2):
    """(Int tau1 ds, Int tau2 ds, Int s tau1 ds, Int s tau2 ds) over s in [0, inf), shape (12,)."""
    scale_length = abs(x_start)
    out = np.zeros(12)
    tau = np.empty(6)
    for p in range(PANEL_EDGES.size - 1):
        lo, hi = PANEL_EDGES[p], PANEL_EDGES[p + 1]
        half, mid = 0.5 * (hi - lo), 0.5 * (hi + lo)
        for q in range(GL_NODES.size):
            u = mid + half * GL_NODES[q]
            s = scale_length * u / (1.0 - u)
            jac = scale_length / (1.0 - u) ** 2
            w = GL_WEIGHTS[q] * half * jac
            torques_on_line(x_start - s, b, m1, m2, tau)
            for k in range(6):
                out[k] += w * tau[k]
                out[6 + k] += w * s * tau[k]
    return out
