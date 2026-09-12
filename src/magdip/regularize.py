"""Sundman / Poincare time regularization of the ``rho -> 0`` collapse.

The dimensionless potential goes like ``-1/rho**3``, so ``rho ~ (tau_c - tau)**(2/5)``
and ``p_rho ~ rho**(-3/2)`` diverge in finite time.  Introducing a fictitious time ``s``
with ``dtau/ds = g(rho) = rho**q`` (``q = 3/2`` by default) makes ``drho/ds`` bounded.

Implemented as a *Poincare transformation* of the extended Hamiltonian

    Gamma(q, p) = g(rho) * (H(q, p) - E0)

rather than as a bare rescaling of the vector field.  The two differ by the term
``-(H - E0) dg/drho`` in ``dp_rho/ds``, which vanishes on the physical manifold
``H = E0`` but is what keeps the transformed system Hamiltonian -- so a symplectic
integrator stays symplectic in ``s``.  The numerical size of ``Gamma`` is itself a
conservation diagnostic.

The regularized state is 9-dimensional: the 8 physical variables plus the physical
time ``tau`` carried along as ``y[8]``.
"""
from __future__ import annotations

import numpy as np

from . import dynamics as dyn
from .dynamics import IRHO

NDIM_REG = 9
ITAU = 8

DEFAULT_Q = 1.5


def lift(y, tau=0.0):
    """Append physical time to a physical state, giving the 9-dim regularized state."""
    y = np.asarray(y, dtype=float)
    out = np.empty((NDIM_REG,) + y.shape[1:])
    out[:dyn.NDIM] = y
    out[ITAU] = tau
    return out


def drop(z):
    """Strip the time component, returning the 8-dim physical state."""
    return np.asarray(z)[:dyn.NDIM]


def physical_time(z):
    return np.asarray(z)[ITAU]


def make_rhs(E0, q=DEFAULT_Q):
    """Return ``f(s, z)`` for the Poincare-transformed system at energy ``E0``.

    ``E0`` may be a scalar or, for batches, an array broadcasting against the columns
    of ``z``.
    """
    E0 = np.asarray(E0, dtype=float)

    def f(s, z):
        z = np.asarray(z)
        y = z[:dyn.NDIM]
        rho = y[IRHO]
        g = rho ** q
        dgdrho = q * rho ** (q - 1.0)
        dH = E0 * 0.0 + dyn.energy(y) - E0   # H - E0, ~0 on the physical manifold

        out = np.empty_like(z, dtype=float)
        out[:dyn.NDIM] = g * dyn.rhs(s, y)
        out[dyn.IPR] -= dH * dgdrho
        out[ITAU] = g
        return out

    return f


def gamma(z, E0, q=DEFAULT_Q):
    """The extended Hamiltonian ``Gamma = g(rho) (H - E0)``; should stay ~0."""
    y = np.asarray(z)[:dyn.NDIM]
    return y[IRHO] ** q * (dyn.energy(y) - E0)
