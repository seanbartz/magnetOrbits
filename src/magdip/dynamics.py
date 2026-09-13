"""Dimensionless equations of motion for two free planar magnetic dipoles.

Units (see notes/derivation.md): lengths in ``a = sqrt(I/mu) = sqrt(2I/m)``, energies
in ``C/a**3`` with ``C = mu0 * m_mag**2 / (2*pi)``, times in ``sqrt(mu a**5 / C)``.
In these units the reduced mass, the moment of inertia and the coupling are all 1 and
the Hamiltonian contains *no* free parameters::

    H = p_rho**2/2 + L**2/(2 rho**2) + (s1**2 + s2**2)/2 - W(a1, a2)/(2 rho**3)

State vectors are stored **state-first**: shape ``(8,)`` for a single trajectory or
``(8, N)`` for a batch, so the same functions serve `scipy.integrate.solve_ivp` and the
vectorised integrators.
"""
from __future__ import annotations

import numpy as np

NDIM = 8
#: indices into the state vector
IRHO, ITHETA, IPHI1, IPHI2, IPR, IL, IS1, IS2 = range(NDIM)

STATE_NAMES = ("rho", "theta", "phi1", "phi2", "p_rho", "L", "s1", "s2")


def shape_factor(a1, a2):
    """``W(a1, a2) = 2 cos a1 cos a2 - sin a1 sin a2``.

    The full orientational content of the dipole-dipole interaction, with
    ``a_i = phi_i - theta`` the moment angles relative to the line of centres.
    ``W = 2`` is the nose-to-tail "locked" configuration.
    """
    return 2.0 * np.cos(a1) * np.cos(a2) - np.sin(a1) * np.sin(a2)


def shape_factor_grad(a1, a2):
    """Return ``(dW/da1, dW/da2)``."""
    s1, c1 = np.sin(a1), np.cos(a1)
    s2, c2 = np.sin(a2), np.cos(a2)
    return (-2.0 * s1 * c2 - c1 * s2, -2.0 * c1 * s2 - s1 * c2)


def potential(rho, a1, a2):
    """Dimensionless interaction energy ``U = -W/(2 rho**3)``."""
    return -shape_factor(a1, a2) / (2.0 * rho ** 3)


def rhs(t, y):
    """Right-hand side ``dy/dt`` of the dimensionless equations of motion.

    ``t`` is ignored (autonomous); it is present for `solve_ivp` compatibility.
    """
    y = np.asarray(y)
    rho = y[IRHO]
    theta = y[ITHETA]
    p_rho = y[IPR]
    L = y[IL]
    s1 = y[IS1]
    s2 = y[IS2]

    a1 = y[IPHI1] - theta
    a2 = y[IPHI2] - theta
    W = shape_factor(a1, a2)
    W1, W2 = shape_factor_grad(a1, a2)

    inv_r2 = 1.0 / (rho * rho)
    inv_r3 = inv_r2 / rho
    inv_r4 = inv_r3 / rho

    out = np.empty_like(y, dtype=float)
    out[IRHO] = p_rho
    out[ITHETA] = L * inv_r2
    out[IPHI1] = s1
    out[IPHI2] = s2
    out[IPR] = L * L * inv_r3 - 1.5 * W * inv_r4
    out[IL] = -0.5 * (W1 + W2) * inv_r3
    out[IS1] = 0.5 * W1 * inv_r3
    out[IS2] = 0.5 * W2 * inv_r3
    return out


def energy(y):
    """Total dimensionless energy ``H``."""
    y = np.asarray(y)
    rho = y[IRHO]
    a1 = y[IPHI1] - y[ITHETA]
    a2 = y[IPHI2] - y[ITHETA]
    kin = 0.5 * (y[IPR] ** 2 + (y[IL] / rho) ** 2 + y[IS1] ** 2 + y[IS2] ** 2)
    return kin + potential(rho, a1, a2)


def total_angular_momentum(y):
    """``J = L + s1 + s2`` -- the conserved angular momentum."""
    y = np.asarray(y)
    return y[IL] + y[IS1] + y[IS2]


def orbital_angular_momentum(y):
    """``L`` -- *not* conserved; exchanged with spin by the non-central force."""
    return np.asarray(y)[IL]


def spin_angular_momentum(y):
    """``S = s1 + s2``."""
    y = np.asarray(y)
    return y[IS1] + y[IS2]


def cartesian_relative(y):
    """Relative position and velocity ``(x, y, vx, vy)`` from the polar state."""
    y = np.asarray(y)
    rho, theta, p_rho, L = y[IRHO], y[ITHETA], y[IPR], y[IL]
    c, s = np.cos(theta), np.sin(theta)
    thdot = L / rho ** 2
    return (rho * c, rho * s,
            p_rho * c - rho * thdot * s,
            p_rho * s + rho * thdot * c)
