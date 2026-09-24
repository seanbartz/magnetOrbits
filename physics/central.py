"""Analytic / quadrature baseline for the central potential U = -kappa / r^3.

Place in the program
--------------------
Independent oracle for the full dipole code: it shares NO code with model.py/solve.py.
Used by tests (stages 2-3) and later as the reference curve that orientation effects are
measured against.

When the full problem reduces to this
-------------------------------------
* kappa = 1: moments along +z and -z, motion in the xy plane. Torques vanish identically
  and the force is -3/r^4 r_hat, so the reduction is exact.
* kappa = 2: moments locked head-to-tail along r_hat (the "power-law" model of
  arXiv:2212.00876, Eq. 18). Not an exact solution of the free problem; it is the
  adiabatic (E -> 0) limit.

Equations (dimensionless, reduced mass 1, E = v_inf^2 / 2, angular momentum l = b v_inf)
-----------------------------------------------------------------------------------------
    U_eff(r) = l^2/(2 r^2) - kappa/r^3,   barrier at r* = 3 kappa / l^2,
    U_eff(r*) = l^6 / (54 kappa^2).

Collision means reaching r = r_c = 2. It happens iff E > max_{r >= r_c} U_eff(r):

    b_c = (27 kappa^2 / 4)^(1/6) E^(-1/3)     if E < kappa/16   (barrier-limited, r* >= r_c)
    b_c = sqrt(r_c^2 + kappa/(E r_c))        if E > kappa/16   (contact-limited)

(the two branches meet at E = kappa/16, b_c = 2 sqrt(3), for r_c = 2).

Deflection angle (classical orbit integral, Goldstein Eq. 3.97 / Landau-Lifshitz §18),
signed so that attraction with b > 0 gives Theta < 0:

    Theta = pi - 2 Int_{r_min}^inf  b dr / ( r^2 sqrt(1 - b^2/r^2 + kappa/(E r^3)) ).

Numerical method: substitute u = r_min/r, beta = b/r_min, gamma = beta^2 - 1
(= kappa/(E r_min^3), from the turning-point condition). The radicand factorizes exactly,
    1 - beta^2 u^2 + gamma u^3 = (1 - u) Q(u),   Q(u) = 1 + u - gamma u^2,
and u = 1 - s^2 removes the endpoint singularity:

    Theta = pi - 4 beta Int_0^1 ds / sqrt(Q(1 - s^2)),

a smooth integrand evaluated with scipy.integrate.quad. Near orbiting (b -> b_c+ in the
barrier-limited regime) Q(1) -> 0 and Theta diverges logarithmically; the coefficient is
    dTheta / d ln(b - b_c) -> theta_dot(r*) / lambda = 1,
where lambda = sqrt(-U_eff''(r*)) is the barrier instability rate (for 1/r^3 the two rates
coincide exactly).

Large-b limit (impulse approximation, V = -kappa/r^n with n = 3):
    Theta -> -2 kappa / (E b^3).
"""

import numpy as np
from scipy.integrate import quad

CONTACT_SEPARATION = 2.0


def capture_impact_parameter(energy, kappa, contact_separation=CONTACT_SEPARATION):
    """Largest impact parameter that reaches r = contact_separation (both regimes)."""
    rc = contact_separation
    if energy < crossover_energy(kappa, rc):
        return (27.0 * kappa**2 / 4.0) ** (1.0 / 6.0) * energy ** (-1.0 / 3.0)
    return np.sqrt(rc**2 + kappa / (energy * rc))


def crossover_energy(kappa, contact_separation=CONTACT_SEPARATION):
    """Energy at which the barrier radius at critical b equals the contact separation.

    r* = r_c at the critical l^2 = 3 kappa / r_c, and E = l^6 / (54 kappa^2) there.
    """
    return (3.0 * kappa / contact_separation) ** 3 / (54.0 * kappa**2)


def turning_point(energy, impact_parameter, kappa):
    """Outermost turning point r_min: largest positive root of r^3 - b^2 r + kappa/E = 0.

    Returns np.nan if there is none (the particle falls past the barrier).
    """
    roots = np.roots([1.0, 0.0, -impact_parameter**2, kappa / energy])
    real = roots[np.abs(roots.imag) < 1e-12 * max(1.0, impact_parameter)].real
    real = real[real > 0]
    return real.max() if real.size else np.nan


def deflection_angle(energy, impact_parameter, kappa, epsabs=1e-13, epsrel=1e-13):
    """Signed deflection Theta(E, b) for U = -kappa/r^3 (np.nan if no turning point).

    Does not know about contact: callers decide whether r_min < 2 means collision.
    """
    b = impact_parameter
    r_min = turning_point(energy, b, kappa)
    if not np.isfinite(r_min):
        return np.nan
    beta = b / r_min
    gamma = kappa / (energy * r_min**3)

    def integrand(s):
        u = 1.0 - s * s
        return 1.0 / np.sqrt(1.0 + u - gamma * u * u)

    integral, _ = quad(integrand, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel, limit=200)
    return np.pi - 4.0 * beta * integral


def small_angle_deflection(energy, impact_parameter, kappa):
    """Impulse-approximation limit Theta ~ -2 kappa / (E b^3)."""
    return -2.0 * kappa / (energy * impact_parameter**3)
