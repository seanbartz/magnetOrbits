"""Scattering set-up: asymptotic initial conditions, outcomes, observables.

Geometry convention
-------------------
The relative coordinate comes in from ``+x`` moving in the ``-x`` direction with
impact parameter ``b`` offset in ``+y``::

    rho(0) = (sqrt(rho0**2 - b**2),  b),      v(0) = (-v, 0) asymptotically
    L(0)   = +v b,        s1(0) = s2(0) = 0

so the incoming asymptote points along ``-x`` (direction angle ``pi``) and the
deflection angle is ``chi = wrap(psi_out - pi)``.  Orientation angles ``phi1, phi2`` are
lab angles from ``+x``, which is also the asymptote axis, so ``phi_i(0) ~ alpha_i(0)``.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from . import dynamics as dyn
from .dynamics import IRHO, ITHETA, IPHI1, IPHI2, IPR, IL, IS1, IS2

DEFAULT_RHO0 = 200.0
DEFAULT_RHO_CAPTURE = 1.0e-2

#: outcome codes used by the vectorised machinery
RUNNING, ESCAPED, COLLAPSED, TIMEOUT = 0, 1, 2, 3
OUTCOME_NAMES = {RUNNING: "running", ESCAPED: "escape",
                 COLLAPSED: "collapse", TIMEOUT: "timeout"}


def initial_state(E, b, phi1=0.0, phi2=0.0, rho0=DEFAULT_RHO0, s1=0.0, s2=0.0):
    """Asymptotic in-state for the dimensionless system.

    ``E`` is the **translational** (relative-motion) energy, so the asymptotic relative
    speed is ``v = sqrt(2E)`` regardless of the spins.  The total conserved energy is

        H = E + (s1**2 + s2**2)/2 + U(rho0),      U(rho0) = O(rho0**-3)

    so with ``s1 = s2 = 0`` (the default) ``H -> E`` as ``rho0 -> inf``.

    Note that ``phi1, phi2`` are clean *asymptotic* labels only when the corresponding
    spin is zero: a freely spinning dipole has ``phi_i -> s_i * t``, with no limiting
    orientation, so for ``s_i != 0`` the angle is a phase referred to this particular
    ``rho0``.  See :func:`initial_state_total_energy` for the fixed-total-energy variant.

    All arguments broadcast; the result is ``(8,)`` for scalars and ``(8, ...)`` for
    arrays, with the broadcast parameter shape trailing the state axis.

    Raises
    ------
    ValueError
        if ``|b| > rho0`` (no such asymptotic state exists) or ``E <= 0``.
    """
    E, b, phi1, phi2, rho0, s1, s2 = np.broadcast_arrays(
        *[np.asarray(v, dtype=float)
          for v in (E, b, phi1, phi2, rho0, s1, s2)])
    if np.any(E <= 0.0):
        raise ValueError("translational energy E must be positive")
    if np.any(np.abs(b) > rho0):
        raise ValueError("|b| must not exceed rho0: no in-state exists with the "
                         "trajectory starting inside its own impact parameter "
                         "(max |b| = %g, rho0 = %g)"
                         % (float(np.max(np.abs(b))), float(np.min(rho0))))
    v = np.sqrt(2.0 * E)
    x0 = np.sqrt(np.maximum(rho0 ** 2 - b ** 2, 0.0))
    y = np.zeros((dyn.NDIM,) + E.shape)
    y[IRHO] = rho0
    y[ITHETA] = np.arctan2(b, x0)
    y[IPHI1] = phi1
    y[IPHI2] = phi2
    y[IPR] = -v * x0 / rho0
    y[IL] = v * b
    y[IS1] = s1
    y[IS2] = s2
    return y


def initial_state_total_energy(E_total, b, phi1=0.0, phi2=0.0, rho0=DEFAULT_RHO0,
                               s1=0.0, s2=0.0):
    """In-state at fixed *total* asymptotic energy, split between motion and spin.

    The translational share is ``E_total - (s1**2 + s2**2)/2``, which must be positive.
    Use this when comparing spinning and non-spinning in-states at the same ``Etil``.
    """
    spin_ke = 0.5 * (np.asarray(s1, float) ** 2 + np.asarray(s2, float) ** 2)
    E_trans = np.asarray(E_total, float) - spin_ke
    if np.any(E_trans <= 0.0):
        raise ValueError("spin kinetic energy exceeds the total energy")
    return initial_state(E_trans, b, phi1, phi2, rho0, s1, s2)


def asymptotic_parameters(y):
    """Recover ``(rho, E_trans, b)`` from a state, ignoring the spin energy.

    ``E_trans = (p_rho**2 + (L/rho)**2)/2`` and ``b = L/sqrt(2 E_trans)``.  Exact for a
    state built by :func:`initial_state`; for a state at finite ``rho`` it is the
    osculating free-flight equivalent, which is what the flight-time and delay
    bookkeeping needs.
    """
    y = np.asarray(y, dtype=float)
    rho = y[IRHO]
    L = y[IL]
    E_trans = 0.5 * (y[IPR] ** 2 + (L / rho) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where(E_trans > 0.0, L / np.sqrt(2.0 * E_trans), 0.0)
    return rho, E_trans, b


def free_flight_time(E, b, rho0=DEFAULT_RHO0):
    """Time a free particle would take to go from ``rho0`` in to ``rho0`` out."""
    E, b, rho0 = [np.asarray(x, dtype=float) for x in (E, b, rho0)]
    return 2.0 * np.sqrt(np.maximum(rho0 ** 2 - b ** 2, 0.0)) / np.sqrt(2.0 * E)


def wrap_pi(x):
    """Wrap to ``(-pi, pi]``."""
    return -(np.mod(-np.asarray(x) + np.pi, 2.0 * np.pi) - np.pi)


def deflection_angle(y):
    """Deflection ``chi`` of an out-state, relative to the incoming ``-x`` asymptote."""
    _, _, vx, vy = dyn.cartesian_relative(y)
    return wrap_pi(np.arctan2(vy, vx) - np.pi)


def spin_fraction(y):
    """``S/J`` -- fraction of the conserved angular momentum carried by spin."""
    J = dyn.total_angular_momentum(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(J != 0.0, dyn.spin_angular_momentum(y) / J, np.nan)


# ---------------------------------------------------------------------------
# reference single-trajectory integration (scipy; used by tests and for plots)
# ---------------------------------------------------------------------------

def run_trajectory(E, b, phi1=0.0, phi2=0.0, rho0=DEFAULT_RHO0,
                   rho_capture=DEFAULT_RHO_CAPTURE, rho_escape=None,
                   t_max=None, rtol=1e-12, atol=1e-12, method="DOP853",
                   dense_output=False, max_step=np.inf, y0=None, s1=0.0, s2=0.0):
    """Integrate one trajectory with SciPy until it escapes, collapses or times out.

    Returns a dict with the SciPy solution, the outcome code and the observables.
    """
    if y0 is None:
        y0 = initial_state(E, b, phi1, phi2, rho0, s1, s2)
    else:
        # A supplied state is authoritative: re-derive the asymptotic parameters from
        # it so that the free-flight time, the time delay and the timeout horizon
        # describe the trajectory actually being integrated rather than whatever was
        # passed for E and b.
        y0 = np.asarray(y0, dtype=float)
        rho0, E, b = asymptotic_parameters(y0)
    y0 = np.asarray(y0, dtype=float)
    if rho_escape is None:
        rho_escape = rho0
    if t_max is None:
        t_max = 200.0 * free_flight_time(E, b, rho0) + 1000.0

    def ev_capture(t, y):
        return y[IRHO] - rho_capture
    ev_capture.terminal = True
    ev_capture.direction = -1

    def ev_escape(t, y):
        return y[IRHO] - rho_escape * (1.0 + 1e-9)
    ev_escape.terminal = True
    ev_escape.direction = +1

    sol = solve_ivp(dyn.rhs, (0.0, t_max), y0, method=method, rtol=rtol, atol=atol,
                    events=(ev_capture, ev_escape), dense_output=dense_output,
                    max_step=max_step)

    if sol.t_events[0].size:
        outcome, t_end, y_end = COLLAPSED, sol.t_events[0][0], sol.y_events[0][0]
    elif sol.t_events[1].size:
        outcome, t_end, y_end = ESCAPED, sol.t_events[1][0], sol.y_events[1][0]
    else:
        outcome, t_end, y_end = TIMEOUT, sol.t[-1], sol.y[:, -1]

    E0 = dyn.energy(y0)
    return {
        "sol": sol,
        "outcome": outcome,
        "outcome_name": OUTCOME_NAMES[outcome],
        "t_end": t_end,
        "y_end": y_end,
        "chi": deflection_angle(y_end) if outcome == ESCAPED else np.nan,
        "delay": (t_end - free_flight_time(E, b, rho0)) if outcome == ESCAPED else np.nan,
        "E0": E0,
        "J0": dyn.total_angular_momentum(y0),
        "dE": dyn.energy(y_end) - E0,
        "dJ": dyn.total_angular_momentum(y_end) - dyn.total_angular_momentum(y0),
        "spin_fraction": spin_fraction(y_end),
        "L_end": y_end[IL],
        "rho_min": float(np.min(sol.y[IRHO])),
    }


# ---------------------------------------------------------------------------
# vectorised batch scattering
# ---------------------------------------------------------------------------

def _events(rho_capture, rho_escape):
    from .integrators import Event
    return [Event(lambda t, y: y[IRHO] - rho_capture, -1, COLLAPSED, "collapse"),
            Event(lambda t, y: y[IRHO] - rho_escape * (1.0 + 1e-9), +1, ESCAPED,
                  "escape")]


def run_batch(E, b, phi1=0.0, phi2=0.0, rho0=DEFAULT_RHO0,
              rho_capture=DEFAULT_RHO_CAPTURE, rho_escape=None,
              t_max=None, rtol=1e-11, atol=1e-12, regularized=False,
              q=None, max_steps=2_000_000, progress=None, s1=0.0, s2=0.0):
    """Integrate a batch of scattering trajectories and return their observables.

    ``E, b, phi1, phi2, s1, s2`` broadcast against each other; ``E`` is the
    translational energy (see :func:`initial_state`).  Any broadcast shape is accepted
    -- including two-dimensional sweeps such as ``phi1[:, None], phi2[None, :]`` -- and
    every per-trajectory output is returned flat, with the broadcast shape under the
    ``"shape"`` key so results can be reshaped (see :func:`reshape_results`).  With ``regularized=True`` the
    integration is done in Sundman time (see `magdip.regularize`), which is much
    cheaper for trajectories that approach collapse but costs a little accuracy on
    wide, fast fly-bys.
    """
    from . import integrators as itg
    from . import regularize as reg

    y0 = initial_state(E, b, phi1, phi2, rho0, s1, s2)
    param_shape = y0.shape[1:]
    # any broadcast shape is flattened to a single column axis for the integrator;
    # `param_shape` is returned so a multi-dimensional sweep can be reshaped back
    y0 = y0.reshape(dyn.NDIM, -1)
    N = y0.shape[1]
    Eb = np.broadcast_to(np.asarray(E, float), param_shape).reshape(-1)
    bb = np.broadcast_to(np.asarray(b, float), param_shape).reshape(-1)
    if N == 1 and Eb.size == 0:      # all-scalar inputs
        Eb = np.atleast_1d(np.asarray(E, float))
        bb = np.atleast_1d(np.asarray(b, float))

    if rho_escape is None:
        rho_escape = rho0
    t_free = free_flight_time(Eb, bb, rho0)
    if t_max is None:
        t_max = 200.0 * t_free + 1000.0

    E0 = dyn.energy(y0)
    J0 = dyn.total_angular_momentum(y0)

    if regularized:
        q = reg.DEFAULT_Q if q is None else q
        z0 = reg.lift(y0)
        # per-trajectory horizons, sliced alongside E0 when columns are retired, so a
        # trajectory's classification never depends on which others share its batch
        state = {"E0": E0, "tau_max": np.broadcast_to(t_max, (N,)).astype(float)}

        def build():
            f = reg.make_rhs(state["E0"], q=q)
            evs = [itg.Event(lambda t, z: z[IRHO] - rho_capture, -1, COLLAPSED),
                   itg.Event(lambda t, z: z[IRHO] - rho_escape * (1.0 + 1e-9), +1,
                             ESCAPED),
                   itg.Event(lambda t, z: z[reg.ITAU] - state["tau_max"], +1, TIMEOUT)]
            return f, evs

        def rebind(keep):
            state["E0"] = state["E0"][keep]
            state["tau_max"] = state["tau_max"][keep]
            return build()

        f, evs = build()
        res = itg.integrate_batch(f, z0, np.inf, evs, rtol=rtol, atol=atol,
                                  max_steps=max_steps, progress=progress,
                                  rebind=rebind)
        y_end = reg.drop(res["y"])
        t_end = reg.physical_time(res["y"])
    else:
        res = itg.integrate_batch(dyn.rhs, y0, t_max, _events(rho_capture, rho_escape),
                                  rtol=rtol, atol=atol, max_steps=max_steps,
                                  progress=progress)
        y_end = res["y"]
        t_end = res["t"]

    outcome = res["outcome"].astype(np.int8)
    escaped = outcome == ESCAPED
    chi = np.where(escaped, deflection_angle(y_end), np.nan)
    delay = np.where(escaped, t_end - t_free, np.nan)

    return {
        "y0": y0, "y_end": y_end, "t_end": t_end,
        "outcome": outcome, "escaped": escaped, "collapsed": outcome == COLLAPSED,
        "chi": chi, "delay": delay,
        "E0": E0, "J0": J0,
        "dE": dyn.energy(y_end) - E0,
        "dJ": dyn.total_angular_momentum(y_end) - J0,
        "spin_fraction": spin_fraction(y_end),
        "L_end": y_end[IL],
        "nsteps": res["nsteps"], "nrejected": res["nrejected"],
        "b": bb, "E": Eb, "shape": param_shape,
    }


def reshape_results(r):
    """Reshape a :func:`run_batch` result's per-trajectory columns to the sweep shape.

    ``run_batch`` always returns flat ``(N,)`` columns; this restores the broadcast
    shape of the inputs, which is what a two-dimensional sweep wants.
    """
    shape = tuple(r["shape"])
    if not shape:
        return r
    flat = ("outcome", "escaped", "collapsed", "chi", "delay", "t_end", "E0", "J0",
            "dE", "dJ", "spin_fraction", "L_end", "nsteps", "nrejected", "b", "E")
    out = dict(r)
    for k in flat:
        if k in out:
            out[k] = np.asarray(out[k]).reshape(shape)
    return out
