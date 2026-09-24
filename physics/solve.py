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
are exactly E and b sqrt(2E). Moments m1, m2 and spins w1, w2 are given at r0 in the lab
frame. Remaining finite-r0 effects, checked for convergence:
  * the velocity direction at r0 ignores the bending accumulated on (-inf, r0), O(b/E r0^4);
  * for a non-spinning in-state the torque ~ 1/r^3 has already turned the moments by
    O(1 / (I E r0)) when coming from infinity; irrelevant when torques vanish (+z/-z),
    handled in a later stage otherwise.

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
from scipy.integrate import solve_ivp

from physics import observables
from physics.model import (CONTACT_SEPARATION, MIN_RADIAL_FORCE_COEFF, SphereParams,
                           force_on_2, potential, torques)

# Escape criterion coefficient: r'^2/2 > ESCAPE_COEFF / r^3, with
# ESCAPE_COEFF = Int_r^inf 6/r'^4 dr' * r^3 = 2.
ESCAPE_COEFF = -MIN_RADIAL_FORCE_COEFF / 3.0


def initial_state(energy, impact_parameter, m1, m2, w1=(0, 0, 0), w2=(0, 0, 0), start_radius=1000.0):
    """19-component in-state (see module docstring). Moments are normalized here.

    The speed v0 at r0 is chosen so that v0^2/2 + U(r0) = E exactly (the asymptotic
    translational energy), and the lateral offset b0 = b v_inf / v0 so that the orbital
    angular momentum equals b sqrt(2E) exactly. Without this, H(r0) = E + U(r0) and the
    deflection carries an O(1/r0^3) error (measured: 3e-8 at E = 0.03, b = 5, r0 = 1000).
    """
    if impact_parameter >= start_radius:
        raise ValueError("impact parameter must be smaller than the start radius")
    m1 = np.asarray(m1, dtype=float) / np.linalg.norm(m1)
    m2 = np.asarray(m2, dtype=float) / np.linalg.norm(m2)
    speed_at_infinity = np.sqrt(2.0 * energy)

    # Fixed-point iteration: U(r0) depends (weakly, via the offset) on b0, which depends on v0.
    offset = impact_parameter
    for _ in range(3):
        position = np.array([-np.sqrt(start_radius**2 - offset**2), offset, 0.0])
        speed = np.sqrt(2.0 * (energy - potential(position, m1, m2)))
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


def run_trajectory(energy, impact_parameter, m1, m2, w1=(0, 0, 0), w2=(0, 0, 0),
                   params=SphereParams(), start_radius=1000.0, outgoing_radius=None,
                   stop="r_out", rtol=1e-10, atol=1e-12, max_time=None,
                   fixed_centers=False, keep_solution=False):
    """Integrate one trajectory and summarize it.

    Parameters
    ----------
    energy, impact_parameter : E_tilde, b_tilde of the in-state.
    m1, m2 : initial moment directions (3,), lab frame; normalized internally.
    w1, w2 : initial spin angular velocities (3,).
    params : SphereParams (inertia factor beta).
    start_radius : r0 of the in-state.
    outgoing_radius : r_out where an escaping run stops (default = start_radius).
    stop : "r_out" (asymptotic deflection) or "escape" (stop at the rigorous criterion).
    rtol, atol : solve_ivp tolerances (atol applies to every component).
    max_time : timeout; default 4 r0 / v_inf + 2000.
    fixed_centers : freeze r (rotational dynamics only).
    keep_solution : also return the solve_ivp result (dense output on).

    Returns dict with
        outcome            "contact" | "escape" | "timeout"
        t_end, y_end       end time and state (for contact: state at first touch)
        r_min              smallest periapsis distance found (or r at contact)
        deflection         signed planar deflection with windings (nan if non-planar/not escaped)
        deflection_3d      unsigned angle between initial and final velocity
        energy_drift, angular_momentum_drift, moment_norm_error : validation numbers
        n_rhs, escape_time, params/settings echo
    """
    inertia = params.inertia_tilde
    y0 = initial_state(energy, impact_parameter, m1, m2, w1, w2, start_radius)
    speed = np.sqrt(2.0 * energy)
    if max_time is None:
        max_time = 4.0 * start_radius / speed + 2000.0
    if outgoing_radius is None:
        outgoing_radius = start_radius

    if stop not in ("r_out", "escape"):
        raise ValueError(f"unknown stop mode {stop!r}")
    events = [_contact_event, _periapsis_event, _make_escape_event(stop == "escape"),
              _make_outgoing_radius_event(outgoing_radius)]
    if fixed_centers:
        events = []

    sol = solve_ivp(equations_of_motion, (0.0, max_time), y0, method="DOP853",
                    rtol=rtol, atol=atol, events=events, dense_output=keep_solution,
                    args=(inertia, fixed_centers))
    if sol.status == -1:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    # 1. Classify the outcome.
    y_end, t_end = sol.y[:, -1], sol.t[-1]
    outcome, r_min, escape_time = "timeout", np.nan, np.nan
    if not fixed_centers:
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
                y_end = y_contact[0]
            else:
                y_end = y_peri[grazing[0]]
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
        "outcome": outcome, "t_end": t_end, "y_end": y_end, "y0": y0, "r_min": r_min,
        "deflection": deflection, "deflection_3d": deflection_3d, "escape_time": escape_time,
        "energy_initial": energy_series[0],
        "energy_drift": observables.relative_drift(energy_series),
        "angular_momentum_drift": observables.relative_drift(j_series),
        "moment_norm_error": observables.moment_norm_error(ys),
        "n_rhs": sol.nfev,
        "settings": dict(energy=energy, impact_parameter=impact_parameter, start_radius=start_radius,
                         outgoing_radius=outgoing_radius, stop=stop, rtol=rtol, atol=atol,
                         max_time=max_time, fixed_centers=fixed_centers,
                         inertia_factor=params.inertia_factor),
    }
    if keep_solution:
        result["solution"] = sol
    return result
