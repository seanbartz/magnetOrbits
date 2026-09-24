"""Dimensional, Cartesian, lab-frame reference implementation.

Deliberately written from the *vector* force and torque expressions rather than from
the polar Lagrangian, so that it is an independent check on `magdip.dynamics`:

    U    = -(mu0/4pi r^3) [ 3 (m1.rhat)(m2.rhat) - m1.m2 ]
    F1   = (3 mu0/4pi r^4) [ (m1.m2) rhat - 5 (m1.rhat)(m2.rhat) rhat
                             + (m1.rhat) m2 + (m2.rhat) m1 ]
    tau1 = m1 x B2,   B2 = (mu0/4pi r^3) [ 3 (m2.rhat) rhat - m2 ]

with ``r = r1 - r2``.  Two particles, planar, 12-dimensional state

    [x1, y1, x2, y2, phi1, phi2, vx1, vy1, vx2, vy2, w1, w2]

This module is used only by the test suite and for demonstrations; production runs use
the dimensionless system.
"""
from __future__ import annotations

import numpy as np

MU0 = 4.0e-7 * np.pi

#: electron gyromagnetic ratio, rad/(s T) -- only used to *size the term we neglect*
GAMMA_E = 1.76085963e11


class DipolePair(object):
    """Two identical planar point dipoles with explicit physical parameters."""

    def __init__(self, mass, inertia, moment, mu0=MU0, radius=None):
        self.mass = float(mass)
        self.inertia = float(inertia)
        self.moment = float(moment)
        self.mu0 = float(mu0)
        #: physical body radius, if the magnet is an extended object.  Only used to
        #: report where the point-dipole model stops applying; never enters the EOM.
        self.radius = None if radius is None else float(radius)

    @classmethod
    def from_sphere(cls, mass, radius, moment, beta=2.0 / 5.0, mu0=MU0):
        """A uniformly magnetized sphere: ``I = beta m R**2``, ``beta = 2/5`` solid.

        Edwards et al. (2017) show such a sphere is *exactly* a point dipole for the
        external field and force, which is what licenses the model used here.
        """
        return cls(mass, beta * mass * radius ** 2, moment, mu0=mu0, radius=radius)

    # -- rigid-rotor modelling assumption -------------------------------------
    @property
    def intrinsic_spin(self):
        """Angular momentum carried by the *magnetization itself*, ``m/gamma_e``.

        The model in `magdip.dynamics` gives each magnet exactly one rotational degree
        of freedom, with angular momentum ``s = I phi_dot`` -- i.e. all of it comes from
        the angular velocity of the moment's orientation.  A real ferromagnet also
        carries the electron-spin angular momentum responsible for its magnetization
        (Einstein--de Haas / Barnett), which this model deliberately excludes.

        This property exists so that the size of the neglected term is documented and
        checkable rather than silent.  For the 5 mm NdFeB spheres of arXiv:2212.00876 it
        is ~2e-8 in dimensionless units, against spins of order 0.1-1 in the scans.

        Neglecting it is also justified structurally: in a hard magnet the magnetization
        is pinned to the crystal axes, so it co-rotates with the body and there is no
        relative rotation to exchange angular momentum with.
        """
        return self.moment / GAMMA_E

    @property
    def intrinsic_spin_dimensionless(self):
        """:attr:`intrinsic_spin` in units of the problem's angular-momentum scale."""
        return self.intrinsic_spin / self.momentum_scale

    @property
    def contact_radius(self):
        """Dimensionless separation ``2R/a`` at which two such bodies touch.

        ``None`` unless a ``radius`` was supplied.  With ``I = beta m R**2`` this is
        ``sqrt(2/beta)``, independent of size and density: **exactly ``sqrt(5)`` for
        solid spheres**.  The point-dipole treatment, and every conclusion about the
        ``rho -> 0`` collapse, applies only outside this radius.
        """
        if self.radius is None:
            return None
        return 2.0 * self.radius / self.length_scale

    # -- derived scales -------------------------------------------------------
    @property
    def C(self):
        """``C = mu0 m^2 / (2 pi)`` -- the coupling used throughout."""
        return self.mu0 * self.moment ** 2 / (2.0 * np.pi)

    @property
    def reduced_mass(self):
        return self.mass / 2.0

    @property
    def length_scale(self):
        """``a = sqrt(I/mu)`` -- radius of gyration, the unique length."""
        return np.sqrt(self.inertia / self.reduced_mass)

    @property
    def energy_scale(self):
        return self.C / self.length_scale ** 3

    @property
    def time_scale(self):
        return np.sqrt(self.reduced_mass * self.length_scale ** 2 / self.energy_scale)

    @property
    def momentum_scale(self):
        """Unit of angular momentum, ``mu a^2 / t0``."""
        return self.reduced_mass * self.length_scale ** 2 / self.time_scale

    # -- interaction ----------------------------------------------------------
    def _vectors(self, state):
        r = np.array([state[0] - state[2], state[1] - state[3]])
        rmag = np.hypot(r[0], r[1])
        rhat = r / rmag
        m1 = self.moment * np.array([np.cos(state[4]), np.sin(state[4])])
        m2 = self.moment * np.array([np.cos(state[5]), np.sin(state[5])])
        return r, rmag, rhat, m1, m2

    def potential(self, state):
        _, rmag, rhat, m1, m2 = self._vectors(state)
        k = self.mu0 / (4.0 * np.pi * rmag ** 3)
        return -k * (3.0 * m1.dot(rhat) * m2.dot(rhat) - m1.dot(m2))

    def force_on_1(self, state):
        """``F1``; the force on particle 2 is ``-F1`` (Newton's third law)."""
        _, rmag, rhat, m1, m2 = self._vectors(state)
        k = 3.0 * self.mu0 / (4.0 * np.pi * rmag ** 4)
        m1r, m2r = m1.dot(rhat), m2.dot(rhat)
        return k * (m1.dot(m2) * rhat - 5.0 * m1r * m2r * rhat + m1r * m2 + m2r * m1)

    def field_torques(self, state):
        """Field torques ``(m1 x B2, m2 x B1)`` -- the z-components."""
        _, rmag, rhat, m1, m2 = self._vectors(state)
        k = self.mu0 / (4.0 * np.pi * rmag ** 3)
        B2 = k * (3.0 * m2.dot(rhat) * rhat - m2)
        B1 = k * (3.0 * m1.dot(rhat) * rhat - m1)
        cross = lambda u, v: u[0] * v[1] - u[1] * v[0]
        return cross(m1, B2), cross(m2, B1)

    # -- dynamics -------------------------------------------------------------
    def rhs(self, t, state):
        F1 = self.force_on_1(state)
        t1, t2 = self.field_torques(state)
        out = np.empty(12)
        out[0:4] = state[6:10]
        out[4] = state[10]
        out[5] = state[11]
        out[6:8] = F1 / self.mass
        out[8:10] = -F1 / self.mass
        out[10] = t1 / self.inertia
        out[11] = t2 / self.inertia
        return out

    def energy(self, state):
        ke = 0.5 * self.mass * (state[6] ** 2 + state[7] ** 2 + state[8] ** 2 + state[9] ** 2)
        ke += 0.5 * self.inertia * (state[10] ** 2 + state[11] ** 2)
        return ke + self.potential(state)

    def total_angular_momentum(self, state):
        """``J`` about the origin: orbital + spin."""
        orb = self.mass * ((state[0] * state[7] - state[1] * state[6])
                           + (state[2] * state[9] - state[3] * state[8]))
        return orb + self.inertia * (state[10] + state[11])

    # -- mapping to the dimensionless system ----------------------------------
    def to_dimensionless(self, state):
        """Map a lab Cartesian state to the dimensionless polar state ``(8,)``.

        Assumes the centre of mass is at rest at the origin (the CM frame).
        """
        a, t0 = self.length_scale, self.time_scale
        rx, ry = state[0] - state[2], state[1] - state[3]
        vx, vy = state[6] - state[8], state[7] - state[9]
        rmag = np.hypot(rx, ry)
        theta = np.arctan2(ry, rx)
        rho = rmag / a
        # radial and angular components, then convert to dimensionless momenta
        p_rho = (rx * vx + ry * vy) / rmag * (t0 / a)
        L = (rx * vy - ry * vx) * (t0 / a ** 2)
        return np.array([rho, theta, state[4], state[5], p_rho, L,
                         state[10] * t0, state[11] * t0])

    def from_dimensionless(self, y):
        """Inverse of :meth:`to_dimensionless`, in the CM frame."""
        a, t0 = self.length_scale, self.time_scale
        rho, theta, phi1, phi2, p_rho, L, s1, s2 = y
        r = rho * a
        thdot = L / rho ** 2 / t0
        c, s = np.cos(theta), np.sin(theta)
        rdot = p_rho * a / t0
        vx = rdot * c - r * thdot * s
        vy = rdot * s + r * thdot * c
        state = np.zeros(12)
        state[0:2] = (0.5 * r * c, 0.5 * r * s)
        state[2:4] = (-0.5 * r * c, -0.5 * r * s)
        state[4], state[5] = phi1, phi2
        state[6:8] = (0.5 * vx, 0.5 * vy)
        state[8:10] = (-0.5 * vx, -0.5 * vy)
        state[10], state[11] = s1 / t0, s2 / t0
        return state
