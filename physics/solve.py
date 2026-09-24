"""Integrate one scattering trajectory of two free magnetic spheres.

Place in the program
--------------------
    run_single.py / sweep.py / tests
        -> solve.run_trajectory(...)            (THE one code path for every run)
            -> solve.initial_state(...)
            -> scipy.integrate.solve_ivp(solve.equations_of_motion, events=...)
                -> model.force_on_2, model.torques
            -> observables.* (outcome summary)

Equations (dimensionless; scales and derivation in model.py)
------------------------------------------------------------
    r'  = v                       v'  = F2(r, m1, m2)          (fixed_centers: both 0)
    m1' = w1 x m1                 w1' = tau_1(r, m1, m2) / I
    m2' = w2 x m2                 w2' = tau_2(r, m1, m2) / I
    psi' = (x v_y - y v_x) / (x^2 + y^2)       (swept azimuth; bookkeeping only)

State layout (19,): [r | v | m1 | m2 | w1 | w2 | psi]  (see observables.SLICES).
A sphere's inertia tensor is isotropic, so the moment direction m_i is the only
orientational degree of freedom that the dynamics sees; no quaternions are needed.

In-state
--------
Relative particle at distance r0 on the line y = b0, z = 0, moving along +x:
r = (-sqrt(r0^2 - b0^2), b0, 0), v = (v0, 0, 0), with v0^2/2 + U(r0) = E and
b0 v0 = b sqrt(2E), so the translational energy and orbital angular momentum at infinity
are exactly E and b sqrt(2E). Moments m1, m2 are lab-frame directions at t -> -inf
(non-spinning in-state): the torque, generically ~ 1/r^3, has already turned them by
O(1/(I E r0)) and spun them up by O(1/(I v r0^2)) before r0, and the first-order values of
both are added (incoming_leg_correction; switch off with asymptotic_correction=False, in
which case m and w refer to r0). Remaining finite-r0 effects, checked for convergence:
  * the velocity direction at r0 ignores the bending accumulated on (-inf, r0), O(b/E r0^4);
  * second-order orientation terms and the O(I w) orbital angular momentum handed to the
    spins on the incoming leg, both O(r0^-2).

Numerical method
----------------
scipy.integrate.solve_ivp with DOP853 (8th-order explicit Runge-Kutta, adaptive).
Chosen over a symplectic scheme because trajectories are scattering (finite time near
the interaction region, no long-time Hamiltonian evolution) and because event location
(contact, escape) with dense output is built in. Expected behavior: energy and |m_i|
drift at roughly the rtol level, growing ~linearly with the number of steps. The
unit-vector constraint |m_i| = 1 is NOT enforced; its residual is reported.

Outcomes and events
-------------------
  "contact"  r reaches 2 (spheres touch). Terminal. Because solve_ivp only sees sign
             changes of r - 2 at step boundaries, a grazing dip below 2 inside one step
             could be missed; every periapsis (r.v crossing 0 upward) is therefore also
             located, and a periapsis with r < 2 counts as contact at that time.
  "escape"   rigorous: r' > 0 and r'^2/2 > 2/r^3. Since the radial dipole force is
             >= -6/r^4 for all orientations (model.MIN_RADIAL_FORCE_COEFF) and the
             centrifugal term is >= 0, r' can then never return to zero.
             With stop="escape" the run ends here (cheap: for outcome maps);
             with stop="r_out" (default) the run continues to r = r_out so the
             deflection angle is asymptotic.
  "timeout"  t_max reached without either (long-lived transient complexes).

Interface
---------
run_trajectory(energy, impact_parameter, m1, m2, ...) -> dict (see its docstring).
"""

import numpy as np
from scipy.integrate import quad_vec, solve_ivp

from physics import observables
from physics.model import (CONTACT_SEPARATION, MIN_RADIAL_FORCE_COEFF, SphereParams,
                           force_on_2, potential, torques)

# Escape criterion coefficient: r'^2/2 > ESCAPE_COEFF / r^3, with
# ESCAPE_COEFF = Int_r^inf 6/r'^4 dr' * r^3 = 2.
ESCAPE_COEFF = -MIN_RADIAL_FORCE_COEFF / 3.0


def rotate_vector(vec, rotation_vector):
    """Rodrigues rotation of vec by the rotation vector theta (axis theta/|theta|, angle |theta|)."""
    angle = np.linalg.norm(rotation_vector)
    if angle == 0.0:
        return vec.copy()
    axis = rotation_vector / angle
    return (vec * np.cos(angle) + np.cross(axis, vec) * np.sin(angle)
            + axis * (axis @ vec) * (1.0 - np.cos(angle)))


def incoming_leg_correction(energy, impact_parameter, x_start, m1, m2, inertia_tilde):
    """First-order effect of the torques on the incoming leg (-inf, x_start].

    Along the unperturbed straight line r(x) = (x, b, 0), x = x_start - s, speed v = sqrt(2E),
    with the moments frozen at their asymptotic directions m_i:

        w_i(x_start)     = 1/(I v)   Int_0^inf tau_i(x_start - s) ds
        theta_i(x_start) = 1/(I v^2) Int_0^inf s tau_i(x_start - s) ds     (rotation vector)

    Returns (theta1, theta2, w1, w2). Generic torques fall off as 1/r^3, so theta = O(1/(I E r0))
    and w = O(1/(I v r0^2)); neglected terms are O(r0^-2) in theta.
    """
    speed = np.sqrt(2.0 * energy)

    def integrand(s):
        position = np.array([x_start - s, impact_parameter, 0.0])
        tau1, tau2 = torques(position, m1, m2)
        return np.concatenate([tau1, tau2, s * tau1, s * tau2])

    moments, _ = quad_vec(integrand, 0.0, np.inf, epsabs=1e-20, epsrel=1e-11)
    w1 = moments[0:3] / (inertia_tilde * speed)
    w2 = moments[3:6] / (inertia_tilde * speed)
    theta1 = moments[6:9] / (inertia_tilde * speed**2)
    theta2 = moments[9:12] / (inertia_tilde * speed**2)
    return theta1, theta2, w1, w2


def initial_state(energy, impact_parameter, m1, m2, w1=(0, 0, 0), w2=(0, 0, 0), start_radius=1000.0,
                  inertia_tilde=SphereParams().inertia_tilde, asymptotic_correction=True):
    """19-component in-state (see module docstring). Moments are normalized here.

    The speed v0 at r0 is chosen so that v0^2/2 + U(r0) + K_rot(r0) = E exactly (the
    asymptotic translational energy), and the lateral offset b0 = b v_inf / v0 so that the
    orbital angular momentum equals b sqrt(2E) exactly. Without this, H(r0) = E + U(r0) and the
    deflection carries an O(1/r0^3) error (measured: 3e-8 at E = 0.03, b = 5, r0 = 1000).

    asymptotic_correction=True (default): m1, m2 are the directions at t -> -inf of a
    NON-spinning in-state, and the state at r0 includes the first-order rotation and spin
    picked up on the way in (incoming_leg_correction). Without it the moments are the
    directions at r0, which differ from the asymptotic ones by O(1/(I E r0)).
    A spinning in-state has no asymptotic orientation (the phase keeps advancing), so the
    correction requires w1 = w2 = 0.
    """
    if impact_parameter >= start_radius:
        raise ValueError("impact parameter must be smaller than the start radius")
    m1 = np.asarray(m1, dtype=float) / np.linalg.norm(m1)
    m2 = np.asarray(m2, dtype=float) / np.linalg.norm(m2)
    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    speed_at_infinity = np.sqrt(2.0 * energy)

    if asymptotic_correction:
        if np.any(w1) or np.any(w2):
            raise ValueError("asymptotic orientation is undefined for a spinning in-state; "
                             "pass asymptotic_correction=False (moments then refer to r0)")
        x_start = -np.sqrt(start_radius**2 - impact_parameter**2)
        theta1, theta2, w1, w2 = incoming_leg_correction(
            energy, impact_parameter, x_start, m1, m2, inertia_tilde)
        m1, m2 = rotate_vector(m1, theta1), rotate_vector(m2, theta2)
        # Spin energy picked up on the way in came out of the orbital energy.
        acquired_rotational_energy = 0.5 * inertia_tilde * (w1 @ w1 + w2 @ w2)
    else:
        # Spin supplied by the caller is intrinsic to the in-state: E stays translational.
        acquired_rotational_energy = 0.0

    # Fixed-point iteration: U(r0) depends (weakly, via the offset) on b0, which depends on v0.
    offset = impact_parameter
    for _ in range(3):
        position = np.array([-np.sqrt(start_radius**2 - offset**2), offset, 0.0])
        speed = np.sqrt(2.0 * (energy - potential(position, m1, m2) - acquired_rotational_energy))
        offset = impact_parameter * speed_at_infinity / speed

    y0 = np.zeros(observables.STATE_SIZE)
    s = observables.unpack(y0)
    s["r"][:] = position
    s["v"][:] = (speed, 0.0, 0.0)
    s["m1"][:] = m1
    s["m2"][:] = m2
    s["w1"][:] = w1
    s["w2"][:] = w2
    s["psi"][:] = np.arctan2(s["r"][1], s["r"][0])
    return y0


def equations_of_motion(t, y, inertia_tilde, fixed_centers=False):
    """dy/dt for the 19-component state."""
    r_vec, v = y[0:3], y[3:6]
    m1, m2 = y[6:9], y[9:12]
    w1, w2 = y[12:15], y[15:18]
    tau1, tau2 = torques(r_vec, m1, m2)

    dydt = np.empty_like(y)
    if fixed_centers:
        dydt[0:6] = 0.0
    else:
        dydt[0:3] = v
        dydt[3:6] = force_on_2(r_vec, m1, m2)
    dydt[6:9] = np.cross(w1, m1)
    dydt[9:12] = np.cross(w2, m2)
    dydt[12:15] = tau1 / inertia_tilde
    dydt[15:18] = tau2 / inertia_tilde
    rho_sq = r_vec[0] ** 2 + r_vec[1] ** 2
    dydt[18] = (r_vec[0] * v[1] - r_vec[1] * v[0]) / rho_sq if rho_sq > 0 else 0.0
    return dydt


# --- event functions (signatures fixed by solve_ivp) -------------------------------------

def _contact_event(t, y, *args):
    return np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2) - CONTACT_SEPARATION


_contact_event.terminal = True
_contact_event.direction = -1


def _periapsis_event(t, y, *args):
    return y[0] * y[3] + y[1] * y[4] + y[2] * y[5]  # r . v


_periapsis_event.terminal = False
_periapsis_event.direction = 1


def _make_escape_event(terminal):
    def event(t, y, *args):
        r = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)
        radial_velocity = (y[0] * y[3] + y[1] * y[4] + y[2] * y[5]) / r
        # Signed form keeps the function continuous and negative on the way in.
        return 0.5 * radial_velocity * abs(radial_velocity) - ESCAPE_COEFF / r**3
    event.terminal = terminal
    event.direction = 1
    return event


def _make_outgoing_radius_event(outgoing_radius):
    def event(t, y, *args):
        return np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2) - outgoing_radius
    event.terminal = True
    event.direction = 1
    return event


def integrate_state(y0, max_time, params=SphereParams(), rtol=1e-10, atol=1e-12,
                    fixed_centers=False, events=None, dense_output=False):
    """Integrate the equations of motion from an arbitrary 19-component state y0.

    The general-purpose entry point (fixed-center rotor tests, time-reversal checks).
    Scattering runs go through run_trajectory, which calls this with its events.
    fixed_centers=True freezes r and v, leaving only the rotor dynamics at fixed separation
    (an external constraint: J is then NOT conserved, E is). Returns the solve_ivp result.
    """
    sol = solve_ivp(equations_of_motion, (0.0, max_time), np.asarray(y0, dtype=float),
                    method="DOP853", rtol=rtol, atol=atol, events=events,
                    dense_output=dense_output, args=(params.inertia_tilde, fixed_centers))
    if sol.status == -1:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol


def run_trajectory(energy, impact_parameter, m1, m2, w1=(0, 0, 0), w2=(0, 0, 0),
                   params=SphereParams(), start_radius=1000.0, outgoing_radius=None,
                   stop="r_out", rtol=1e-10, atol=1e-12, max_time=None,
                   asymptotic_correction=True, keep_solution=False):
    """Integrate one trajectory and summarize it.

    Parameters
    ----------
    energy, impact_parameter : E_tilde, b_tilde of the in-state.
    m1, m2 : initial moment directions (3,), lab frame; normalized internally. With
        asymptotic_correction (default) they are the directions at t -> -inf; otherwise at r0.
    w1, w2 : initial spin angular velocities (3,); nonzero requires asymptotic_correction=False.
    params : SphereParams (inertia factor beta).
    start_radius : r0 of the in-state.
    outgoing_radius : r_out where an escaping run stops (default = start_radius).
    stop : "r_out" (asymptotic deflection) or "escape" (stop at the rigorous criterion).
    rtol, atol : solve_ivp tolerances (atol applies to every component).
    max_time : timeout; default 4 r0 / v_inf + 2000.
    keep_solution : also return the solve_ivp result (dense output on).

    Returns dict with
        outcome            "contact" | "escape" | "timeout"
        contact_via        "event" (r - 2 sign change) | "periapsis" (grazing dip) | None
        t_end, y_end       end time and state (for contact: state at first touch)
        r_min              smallest periapsis distance found (or r at contact)
        deflection         signed planar deflection with windings (nan if non-planar/not escaped)
        deflection_3d      unsigned angle between initial and final velocity
        energy_drift, angular_momentum_drift, moment_norm_error : validation numbers
        n_rhs, escape_time, params/settings echo
    """
    inertia = params.inertia_tilde
    y0 = initial_state(energy, impact_parameter, m1, m2, w1, w2, start_radius,
                       inertia_tilde=inertia, asymptotic_correction=asymptotic_correction)
    speed = np.sqrt(2.0 * energy)
    if max_time is None:
        max_time = 4.0 * start_radius / speed + 2000.0
    if outgoing_radius is None:
        outgoing_radius = start_radius

    if stop not in ("r_out", "escape"):
        raise ValueError(f"unknown stop mode {stop!r}")
    events = [_contact_event, _periapsis_event, _make_escape_event(stop == "escape"),
              _make_outgoing_radius_event(outgoing_radius)]
    sol = integrate_state(y0, max_time, params=params, rtol=rtol, atol=atol,
                          events=events, dense_output=keep_solution)

    # 1. Classify the outcome.
    y_end, t_end = sol.y[:, -1], sol.t[-1]
    outcome, r_min, escape_time, contact_via = "timeout", np.nan, np.nan, None
    t_contact, y_contact = sol.t_events[0], sol.y_events[0]
    t_peri, y_peri = sol.t_events[1], sol.y_events[1]
    r_peri = np.linalg.norm(y_peri[:, 0:3], axis=1) if len(t_peri) else np.array([])
    grazing = np.nonzero(r_peri < CONTACT_SEPARATION)[0]
    r_min = r_peri.min() if r_peri.size else np.nan

    contact_times = list(t_contact) + [t_peri[k] for k in grazing]
    if contact_times:
        outcome = "contact"
        t_end = min(contact_times)
        if len(t_contact) and t_contact[0] == t_end:
            y_end, contact_via = y_contact[0], "event"
        else:
            y_end, contact_via = y_peri[grazing[0]], "periapsis"
        r_min = min(np.linalg.norm(y_end[0:3]), np.nanmin([r_min, np.inf]))
    elif len(sol.t_events[2]) or len(sol.t_events[3]):
        outcome = "escape"
        escape_time = sol.t_events[2][0] if len(sol.t_events[2]) else np.nan

    # 2. Validation numbers along the accepted steps (up to t_end).
    keep = sol.t <= t_end
    ys = np.column_stack([sol.y[:, keep], y_end])
    energy_series = observables.total_energy(ys, inertia)
    j_series = observables.total_angular_momentum(ys, inertia)

    # 3. Scattering observables.
    deflection, deflection_3d = np.nan, np.nan
    if outcome == "escape":
        deflection, deflection_3d = observables.deflection_angles(y0, y_end)

    result = {
        "outcome": outcome, "contact_via": contact_via, "t_end": t_end, "y_end": y_end, "y0": y0, "r_min": r_min,
        "deflection": deflection, "deflection_3d": deflection_3d, "escape_time": escape_time,
        "energy_initial": energy_series[0],
        "energy_drift": observables.relative_drift(energy_series),
        "angular_momentum_drift": observables.relative_drift(j_series),
        "moment_norm_error": observables.moment_norm_error(ys),
        "n_rhs": sol.nfev,
        "settings": dict(energy=energy, impact_parameter=impact_parameter, start_radius=start_radius,
                         outgoing_radius=outgoing_radius, stop=stop, rtol=rtol, atol=atol,
                         max_time=max_time, inertia_factor=params.inertia_factor,
                         asymptotic_correction=asymptotic_correction),
    }
    if keep_solution:
        result["solution"] = sol
    return result
