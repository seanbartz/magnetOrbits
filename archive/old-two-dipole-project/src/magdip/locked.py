"""The locked-dipole (adiabatic, ``Etil -> 0``) limit: analytics and quadrature.

With the moments pinned nose-to-tail along the line of centres, ``W == 2`` and the
dimensionless problem collapses to the central-force Hamiltonian

    H = p_rho**2/2 + L**2/(2 rho**2) - 1/rho**3

which is the standard "locked-dipole" / Langevin capture model.  Everything here is the
baseline that the full four-degree-of-freedom dynamics is measured against.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import quad

#: ``btil_max = BMAX_COEFF * Etil**(-1/3)``
BMAX_COEFF = (27.0 / 4.0) ** (1.0 / 6.0)


def barrier_radius(L):
    """Location of the centrifugal barrier, ``rho* = 3/L**2``."""
    return 3.0 / np.asarray(L, dtype=float) ** 2


def barrier_height(L):
    """Height of the centrifugal barrier, ``U_eff(rho*) = L**6/54``."""
    return np.asarray(L, dtype=float) ** 6 / 54.0


def L_max(E):
    """Largest orbital angular momentum that still falls to the centre."""
    return (54.0 * np.asarray(E, dtype=float)) ** (1.0 / 6.0)


def b_max(E):
    """Capture impact parameter ``(27/4)**(1/6) * E**(-1/3)``."""
    return BMAX_COEFF * np.asarray(E, dtype=float) ** (-1.0 / 3.0)


def cross_section(E):
    """Planar capture cross section (a length): ``2 b_max``, scaling as ``E**(-1/3)``.

    The familiar ``E**(-2/3)`` is the three-dimensional ``pi b_max**2``.
    """
    return 2.0 * b_max(E)


def max_energy_for_barrier_outside(rho_contact):
    """Largest ``Etil`` for which the centrifugal barrier still sits outside ``rho_contact``.

    The barrier of a marginally captured trajectory is at ``rho* = 3/L_max**2`` with
    ``L_max = (54 Etil)**(1/6)``.  Requiring ``rho* > rho_contact`` gives
    ``Etil < (3/rho_contact)**3 / 54``.  Above this energy two *extended* magnets touch
    before the barrier is ever reached, so the point-dipole capture picture is not
    physically realisable for them (it remains perfectly well posed for the idealised
    point-dipole problem).

    For solid spheres ``rho_contact = sqrt(5)`` and the bound is ``Etil < 0.0447``.
    """
    return (3.0 / np.asarray(rho_contact, dtype=float)) ** 3 / 54.0


def is_captured(E, b):
    """Locked-dipole capture criterion, ``b < b_max(E)``."""
    return np.abs(np.asarray(b, dtype=float)) < b_max(E)


def turning_point(E, b):
    """Smallest positive root ``u = 1/rho`` of ``2u**3 - L**2 u**2 + 2E = 0``.

    Returns ``nan`` when the trajectory is captured (no turning point).
    """
    E = float(E)
    L = np.sqrt(2.0 * E) * abs(float(b))
    roots = np.roots([2.0, -L * L, 0.0, 2.0 * E])
    real = np.sort([r.real for r in roots if abs(r.imag) < 1e-12 * max(1.0, abs(r.real))
                    and r.real > 0.0])
    if real.size == 0:
        return np.nan
    return real[0]


def deflection_angle(E, b):
    """Locked-dipole deflection function, by quadrature.

    ``|chi| = 2 L int_0^{u_t} du / sqrt(2E - L**2 u**2 + 2 u**3) - pi``, evaluated with
    the substitution ``u = u_t (1 - w**2)`` that removes the square-root endpoint
    singularity.  The sign follows `magdip.scatter`'s geometry: ``b > 0`` means ``L > 0``
    and an attractive encounter deflects to positive ``chi``.  Returns ``nan`` for
    captured impact parameters.
    """
    E = float(E)
    b = float(b)
    L = np.sqrt(2.0 * E) * abs(b)
    ut = turning_point(E, b)
    if not np.isfinite(ut):
        return np.nan

    def integrand(w):
        u = ut * (1.0 - w * w)
        f = 2.0 * E - L * L * u * u + 2.0 * u ** 3
        f = max(f, 0.0)
        if f == 0.0:
            return 0.0
        return 2.0 * ut * w / np.sqrt(f)

    val, _ = quad(integrand, 0.0, 1.0, limit=200)
    chi = 2.0 * L * val - np.pi
    return np.sign(b) * chi if b != 0 else chi


def rhs(t, y):
    """Central-force RHS on the reduced state ``[rho, theta, p_rho, L]``."""
    rho, _theta, p_rho, L = y
    return np.array([p_rho, L / rho ** 2, L * L / rho ** 3 - 3.0 / rho ** 4, 0.0])


def energy(y):
    rho, _theta, p_rho, L = y
    return 0.5 * p_rho ** 2 + 0.5 * (L / rho) ** 2 - 1.0 / rho ** 3


def run_trajectory(E, b, rho0=200.0, rho_capture=1e-2, rtol=1e-12, atol=1e-12):
    """Integrate one locked-dipole trajectory on ``[rho, theta, p_rho, L]``.

    Returns ``(outcome, chi)`` with outcome ``"escape"`` or ``"collapse"``; ``chi`` is
    ``nan`` for captures.  Used to validate the analytic barrier results.
    """
    from scipy.integrate import solve_ivp

    v = np.sqrt(2.0 * E)
    x0 = np.sqrt(max(rho0 ** 2 - b ** 2, 0.0))
    y0 = np.array([rho0, np.arctan2(b, x0), -v * x0 / rho0, v * b])

    def ev_cap(t, y):
        return y[0] - rho_capture
    ev_cap.terminal = True
    ev_cap.direction = -1

    def ev_esc(t, y):
        return y[0] - rho0 * (1.0 + 1e-9)
    ev_esc.terminal = True
    ev_esc.direction = +1

    sol = solve_ivp(rhs, (0.0, 1e6), y0, method="DOP853", rtol=rtol, atol=atol,
                    events=(ev_cap, ev_esc))
    if sol.t_events[0].size:
        return "collapse", np.nan
    if not sol.t_events[1].size:
        return "timeout", np.nan
    rho, th, pr, L = sol.y_events[1][0]
    thd = L / rho ** 2
    c, s = np.cos(th), np.sin(th)
    vx, vy = pr * c - rho * thd * s, pr * s + rho * thd * c
    chi = np.arctan2(vy, vx) - np.pi
    return "escape", -(np.mod(-chi + np.pi, 2 * np.pi) - np.pi)
