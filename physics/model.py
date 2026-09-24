"""Dimensionless dipole-dipole interaction between two free magnetic spheres.

Place in the program
--------------------
Lowest layer. `solve.py` builds the equations of motion from `force_on_2` and
`torques`; `observables.py` uses `potential`. Nothing here integrates anything.

Physical system
---------------
Two identical uniformly magnetized spheres: mass m, radius R, dipole moment magnitude mu,
moment of inertia I = beta m R^2 (beta = 2/5 for a solid sphere). A uniformly magnetized
sphere is exactly a point dipole for external fields, forces and torques
(Edwards et al., Am. J. Phys. 85, 130 (2017)). No contact force, no gravity.

Nondimensionalization
---------------------
    length   R
    energy   U0 = mu0 mu^2 / (4 pi R^3)      (dipole energy scale at separation R)
    mass     m/2                              (reduced mass of the relative motion)
    time     t0 = sqrt((m/2) R^2 / U0)

With r = r2 - r1 (dimensionless relative position, pointing from sphere 1 to sphere 2),
r_hat = r/|r|, and unit moment directions m1, m2, the interaction energy is
(Bartz & Shaw, arXiv:2212.00876, Eq. 5)

    U = -(1/r^3) [ 3 (m1.r_hat)(m2.r_hat) - m1.m2 ].

Force on sphere 2 (F1 = -F2), from F2 = -grad_r U (same Eq. 8 of the paper, which labels
it F_1 but whose r_hat convention makes it the force on the second dipole):

    F2 = (3/r^4) [ (m1.r_hat) m2 + (m2.r_hat) m1 + (m1.m2) r_hat - 5 (m1.r_hat)(m2.r_hat) r_hat ].

Field of dipole j at dipole i:  B_j = (1/r^3) [ 3 (mj.r_hat) r_hat - mj ]  (symmetric
under r_hat -> -r_hat), and the magnetic torque on dipole i (paper Eq. 10) is

    tau_i = m_i x B_j = (1/r^3) [ 3 (mj.r_hat)(m_i x r_hat) - m_i x m_j ].

The relative equation of motion is r'' = F2 (reduced mass is the mass unit), and each
sphere's spin obeys I_tilde w_i' = tau_i with the dimensionless moment of inertia

    I_tilde = I / ((m/2) R^2) = 2 beta          (= 4/5 for a solid sphere).

Angular momentum bookkeeping: d(r x p)/dt = r x F2 and tau_1 + tau_2 + r x F2 = 0, so
J = r x p + I_tilde (w1 + w2) is conserved while the orbital and spin parts exchange.

Remaining dimensionless inputs: E_tilde = (relative KE)/U0, b_tilde = b/R, the initial
directions of m1 and m2, and I_tilde. Contact (collision) is at r = 2. For uniformly
magnetized spheres of one material, E_tilde = (3/4) rho v_rel^2 / (mu0 M^2): the sphere
radius drops out (see `physical_to_dimensionless`).

Interfaces
----------
All functions take plain numpy arrays of shape (3,) for vectors and return shape (3,)
(or a float). No broadcasting over trajectories yet; that comes with the sweep stage.

Validity
--------
Point-dipole interaction is exact for uniformly magnetized spheres at any r >= 2 (no
overlap). Rigid magnetization (moment fixed in the body); electron-spin angular momentum
of the magnetization (Einstein-de Haas) is neglected. Non-relativistic, magnetostatic.
"""

from dataclasses import dataclass

import numpy as np

# Separation at which two spheres of radius 1 (in units of R) touch.
CONTACT_SEPARATION = 2.0

# Lower bound of the radial force over all moment orientations:
#   F2 . r_hat = (3/r^4) [ m1.m2 - 3 (m1.r_hat)(m2.r_hat) ] = (3/r^4) m1^T (1 - 3 r_hat r_hat^T) m2,
# and the matrix (1 - 3 r_hat r_hat^T) has eigenvalues (1, 1, -2), so the bilinear form over
# unit vectors is >= -2. Hence F_r >= -6/r^4 (attained head-to-tail). Used by the rigorous
# escape criterion in solve.py.
MIN_RADIAL_FORCE_COEFF = -6.0


@dataclass(frozen=True)
class SphereParams:
    """Dimensionless material parameters of the (identical) spheres.

    inertia_factor : beta in I = beta m R^2 (2/5 solid sphere, 2/3 thin shell).
    """

    inertia_factor: float = 0.4

    @property
    def inertia_tilde(self) -> float:
        """Dimensionless moment of inertia I / ((m/2) R^2) = 2 beta."""
        return 2.0 * self.inertia_factor


def potential(r_vec, m1, m2):
    """Interaction energy U(r, m1, m2) in units of U0 (float)."""
    r = np.sqrt(r_vec @ r_vec)
    r_hat = r_vec / r
    a1 = m1 @ r_hat
    a2 = m2 @ r_hat
    return -(3.0 * a1 * a2 - m1 @ m2) / r**3


def force_on_2(r_vec, m1, m2):
    """Force on sphere 2 (= relative acceleration r''), shape (3,). Force on 1 is minus this."""
    r = np.sqrt(r_vec @ r_vec)
    r_hat = r_vec / r
    a1 = m1 @ r_hat
    a2 = m2 @ r_hat
    bracket = a1 * m2 + a2 * m1 + ((m1 @ m2) - 5.0 * a1 * a2) * r_hat
    return (3.0 / r**4) * bracket


def torques(r_vec, m1, m2):
    """Magnetic torques (tau_1, tau_2) on the two dipoles, each shape (3,)."""
    r = np.sqrt(r_vec @ r_vec)
    r_hat = r_vec / r
    a1 = m1 @ r_hat
    a2 = m2 @ r_hat
    m1_cross_m2 = np.cross(m1, m2)
    inv_r3 = 1.0 / r**3
    tau1 = inv_r3 * (3.0 * a2 * np.cross(m1, r_hat) - m1_cross_m2)
    tau2 = inv_r3 * (3.0 * a1 * np.cross(m2, r_hat) + m1_cross_m2)  # m2 x m1 = -m1 x m2
    return tau1, tau2


# ---------------------------------------------------------------------------
# Unit conversion (the only place physical units appear; used for output only)
# ---------------------------------------------------------------------------

MU0 = 4e-7 * np.pi  # vacuum permeability, T m / A (pre-2019 exact value; 1e-10 rel. from CODATA)


def scales(mass, radius, moment):
    """Physical scales (SI) for spheres of given mass [kg], radius [m], moment [A m^2].

    Returns dict with energy_scale U0 [J], time_scale t0 [s], length_scale R [m],
    velocity_scale R/t0 [m/s].
    """
    energy_scale = MU0 * moment**2 / (4.0 * np.pi * radius**3)
    time_scale = np.sqrt(0.5 * mass * radius**2 / energy_scale)
    return {
        "energy_scale": energy_scale,
        "time_scale": time_scale,
        "length_scale": radius,
        "velocity_scale": radius / time_scale,
    }


def physical_to_dimensionless(mass, radius, moment, v_rel, impact_parameter):
    """Map a physical in-state (relative speed v_rel [m/s], impact parameter [m]) to (E_tilde, b_tilde).

    E_tilde = (1/2)(m/2) v_rel^2 / U0,   b_tilde = b / R.
    """
    s = scales(mass, radius, moment)
    energy_tilde = 0.25 * mass * v_rel**2 / s["energy_scale"]
    return energy_tilde, impact_parameter / radius
