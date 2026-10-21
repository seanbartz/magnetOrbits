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
* kappa = 2 with I_tilde = 0: moments head-to-tail along r_hat with their spin ignored (the
  "power-law" model of arXiv:2212.00876, Eq. 18). This is NOT the locked limit of the free
  problem: moments that stay along r_hat must co-rotate with the line of centers, and their
  spin carries angular momentum. Use the locked_* functions below for the locked law.

Locked head-to-tail law (locked_* functions)
--------------------------------------------
Both moments held along r_hat. Each sphere then spins with omega_i = theta_dot z_hat, so

    J = (r^2 + 2 I_tilde) theta_dot,
    E = r_dot^2/2 + J^2 / (2 (r^2 + 2 I_tilde)) - kappa/r^3,      kappa = 2,

with I_tilde = 2 beta = 4/5 for solid spheres (the same normalization as model.py, where
J = r x r_dot + I_tilde (omega_1 + omega_2) and the spin energy is I_tilde omega_i^2 / 2). The
spin vanishes at infinity (theta_dot -> 0), so J = b sqrt(2E) there. The spin-up takes orbital
angular momentum: the centrifugal term is J^2/(2(r^2 + c)), c = 2 I_tilde, smaller than
J^2/(2 r^2), so the barrier is lower and the collision edge larger than for the spin-free law.
This is the adiabatic (E -> 0) limit of free head-to-tail magnets (validated against the full
dynamics in tests/test_locked_law.py).

    U_eff(r) = J^2 / (2 (r^2 + c)) - kappa/r^3.
At critical J the barrier r* satisfies J^2 = 3 kappa (r*^2 + c)^2 / r*^5 and
    E = U_eff(r*) = kappa (r*^2 + 3c) / (2 r*^5),
which is strictly decreasing in r*, so r*(E) is a unique root. (r^2 + c)^2 / r^5 is strictly
decreasing too, so for each J there is one stationary point: the barrier is the global maximum.

    b_c^2 = 3 kappa (r*^2 + c)^2 / (2 E r*^5)            if E < E_x   (barrier-limited, r* >= r_c)
    b_c^2 = (r_c^2 + c) (1 + kappa / (E r_c^3))           if E > E_x   (contact-limited)
    E_x = kappa (r_c^2 + 3c) / (2 r_c^5)                  (= 0.275 for c = 8/5, r_c = 2)

With c = 0 these reduce to the -kappa/r^3 formulas above.

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
from scipy.optimize import brentq

CONTACT_SEPARATION = 2.0
LOCKED_KAPPA = 2.0
SOLID_SPHERE_INERTIA = 0.8      # I_tilde = 2 beta, beta = 2/5 (kept separate from model.py on purpose)


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


# ----------------------------------------------------------------------------------------------
# Locked head-to-tail law, spin included (see module docstring)
# ----------------------------------------------------------------------------------------------

def locked_crossover_energy(inertia=SOLID_SPHERE_INERTIA, kappa=LOCKED_KAPPA,
                            contact_separation=CONTACT_SEPARATION):
    """Energy at which the critical barrier radius reaches contact: kappa (r_c^2 + 3c) / (2 r_c^5)."""
    rc, c = contact_separation, 2.0 * inertia
    return kappa * (rc**2 + 3.0 * c) / (2.0 * rc**5)


def locked_barrier_radius(energy, inertia=SOLID_SPHERE_INERTIA, kappa=LOCKED_KAPPA):
    """Barrier radius r* at the critical angular momentum: root of kappa (r^2 + 3c)/(2 r^5) = E."""
    c = 2.0 * inertia
    f = lambda r: kappa * (r * r + 3.0 * c) / (2.0 * r**5) - energy
    hi = 1.0
    while f(hi) > 0:
        hi *= 2.0
    lo = hi / 2.0
    while f(lo) < 0:
        lo /= 2.0
    return brentq(f, lo, hi, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=200)


def locked_capture_impact_parameter(energy, inertia=SOLID_SPHERE_INERTIA, kappa=LOCKED_KAPPA,
                                    contact_separation=CONTACT_SEPARATION):
    """Largest colliding impact parameter for locked head-to-tail moments, spin included."""
    rc, c = contact_separation, 2.0 * inertia
    if energy < locked_crossover_energy(inertia, kappa, rc):
        r = locked_barrier_radius(energy, inertia, kappa)
        return np.sqrt(3.0 * kappa * (r * r + c) ** 2 / (2.0 * energy * r**5))
    return np.sqrt((rc**2 + c) * (1.0 + kappa / (energy * rc**3)))


def locked_turning_point(energy, impact_parameter, inertia=SOLID_SPHERE_INERTIA, kappa=LOCKED_KAPPA):
    """Outermost turning point of the locked radial motion (np.nan if the barrier is crossed).

    E = J^2/(2(r^2 + c)) - kappa/r^3 with J^2 = 2 E b^2, i.e. the largest positive root of
    2E r^5 + (2E c - J^2) r^3 + 2 kappa r^2 + 2 kappa c = 0.
    """
    c, j2 = 2.0 * inertia, 2.0 * energy * impact_parameter**2
    roots = np.roots([2.0 * energy, 0.0, 2.0 * energy * c - j2, 2.0 * kappa, 0.0, 2.0 * kappa * c])
    real = roots[np.abs(roots.imag) < 1e-9 * max(1.0, impact_parameter)].real
    real = real[real > 0]
    if not real.size:
        return np.nan
    # polish the root (np.roots is only good to ~1e-12 relative near a double root)
    r = real.max()
    g = lambda x: j2 / (2.0 * (x * x + c)) - kappa / x**3 - energy
    try:
        return brentq(g, r * (1 - 1e-6), r * (1 + 1e-6), xtol=1e-15, rtol=4 * np.finfo(float).eps)
    except ValueError:
        return r


def locked_deflection_angle(energy, impact_parameter, inertia=SOLID_SPHERE_INERTIA, kappa=LOCKED_KAPPA,
                            epsabs=1e-13, epsrel=1e-13):
    """Signed orbital deflection for the locked law (np.nan if no turning point).

    theta_dot = J/(r^2 + c). With u = r_min/r the radicand 2(E - U_eff) factorizes exactly as
    (1 - u) G(u),  G(u) = J^2 r_min^2 (1 + u) / ((r_min^2 + c u^2)(r_min^2 + c)) - 2 kappa (1 + u + u^2)/r_min^3,
    and u = 1 - s^2 gives
        Theta = pi - 4 J r_min Int_0^1 ds / ((r_min^2 + c u^2) sqrt(G(u))).
    With inertia = 0 this equals deflection_angle(energy, b, kappa).
    """
    b = impact_parameter
    r_min = locked_turning_point(energy, b, inertia, kappa)
    if not np.isfinite(r_min):
        return np.nan
    c, j = 2.0 * inertia, b * np.sqrt(2.0 * energy)
    rm2 = r_min * r_min

    def integrand(s):
        u = 1.0 - s * s
        g = j * j * rm2 * (1.0 + u) / ((rm2 + c * u * u) * (rm2 + c)) - 2.0 * kappa * (1.0 + u + u * u) / r_min**3
        return 1.0 / ((rm2 + c * u * u) * np.sqrt(g))

    integral, _ = quad(integrand, 0.0, 1.0, epsabs=epsabs, epsrel=epsrel, limit=200)
    return np.pi - 4.0 * j * r_min * integral

