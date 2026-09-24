"""Conserved quantities, constraint residuals, and scattering observables.

Place in the program
--------------------
Called by solve.py (to summarize a trajectory) and by run_single.py / tests.
Operates on the 19-component state vector defined in solve.py:

    y = [ r (3) | v (3) | m1 (3) | m2 (3) | w1 (3) | w2 (3) | psi (1) ]

r = r2 - r1, v = dr/dt, m_i unit moment directions, w_i spin angular velocities, and
psi the swept azimuth of r about z (auxiliary, for counting windings). Arrays of shape
(19,) or (19, n_times) are accepted.

Quantities (dimensionless, see model.py)
----------------------------------------
    E  = |v|^2/2 + (I/2)(|w1|^2 + |w2|^2) + U(r, m1, m2)          conserved
    J  = r x v + I (w1 + w2)                                       conserved
    L  = r x v                                                     NOT conserved in general
    |m_i| - 1                                                      constraint, should stay 0

Deflection angle
----------------
For planar motion (z = 0) the signed deflection about +z is
    Theta = angle(v_final) - angle(v_initial),  counted with windings,
with the winding number fixed by the swept azimuth psi: for straight asymptotes at
distances r_in, r_out with impact parameters b_in, b_out,
    Theta ~ Delta psi - sign(L_z) [pi - asin(b_in/r_in) - asin(b_out/r_out)].
The velocity-based value (exact mod 2 pi) is shifted by the multiple of 2 pi closest to
this estimate. For non-planar motion only the unsigned angle arccos(v_i . v_f) is defined.
"""

import numpy as np

from physics.model import potential

SLICES = {
    "r": slice(0, 3), "v": slice(3, 6), "m1": slice(6, 9), "m2": slice(9, 12),
    "w1": slice(12, 15), "w2": slice(15, 18), "psi": slice(18, 19),
}
STATE_SIZE = 19


def unpack(y):
    """Dict of views into the state vector (works for shape (19,) or (19, n))."""
    return {name: y[s] for name, s in SLICES.items()}


def _as_columns(y):
    y = np.asarray(y, dtype=float)
    return y[:, None] if y.ndim == 1 else y


def energy_parts(y, inertia_tilde):
    """(translational KE, rotational KE, potential) arrays over time."""
    y = _as_columns(y)
    s = unpack(y)
    kinetic = 0.5 * np.sum(s["v"] ** 2, axis=0)
    rotational = 0.5 * inertia_tilde * (np.sum(s["w1"] ** 2, axis=0) + np.sum(s["w2"] ** 2, axis=0))
    pot = np.array([potential(s["r"][:, k], s["m1"][:, k], s["m2"][:, k]) for k in range(y.shape[1])])
    return kinetic, rotational, pot


def total_energy(y, inertia_tilde):
    kinetic, rotational, pot = energy_parts(y, inertia_tilde)
    return kinetic + rotational + pot


def orbital_angular_momentum(y):
    """L = r x v, shape (3, n)."""
    y = _as_columns(y)
    s = unpack(y)
    return np.cross(s["r"], s["v"], axis=0)


def total_angular_momentum(y, inertia_tilde):
    """J = r x v + I (w1 + w2), shape (3, n)."""
    y = _as_columns(y)
    s = unpack(y)
    return orbital_angular_momentum(y) + inertia_tilde * (s["w1"] + s["w2"])


def moment_norm_error(y):
    """max_t max_i | |m_i| - 1 |  (the unit-vector constraint is not enforced by the ODE solver)."""
    y = _as_columns(y)
    s = unpack(y)
    return max(np.max(np.abs(np.linalg.norm(s["m1"], axis=0) - 1.0)),
               np.max(np.abs(np.linalg.norm(s["m2"], axis=0) - 1.0)))


def relative_drift(series):
    """max_t |Q(t) - Q(0)| / |Q(0)| (vector: norm of the difference); absolute if Q(0) ~ 0."""
    series = np.asarray(series, dtype=float)
    if series.ndim == 1:
        change, reference = np.max(np.abs(series - series[0])), abs(series[0])
    else:
        change = np.max(np.linalg.norm(series - series[:, :1], axis=0))
        reference = np.linalg.norm(series[:, 0])
    return change / reference if reference > 1e-300 else change


def impact_parameter_of(y):
    """|r x v| / |v|: impact parameter of the straight line tangent to the state."""
    s = unpack(np.asarray(y, dtype=float))
    return np.linalg.norm(np.cross(s["r"], s["v"])) / np.linalg.norm(s["v"])


def deflection_angles(y_initial, y_final):
    """(signed planar deflection with windings, unsigned 3D angle) between initial and final velocity.

    The signed value is np.nan when the motion is not planar.
    """
    si, sf = unpack(np.asarray(y_initial)), unpack(np.asarray(y_final))
    vi, vf = si["v"], sf["v"]
    unsigned = np.arccos(np.clip(vi @ vf / (np.linalg.norm(vi) * np.linalg.norm(vf)), -1.0, 1.0))

    planar = (abs(si["r"][2]) < 1e-9 and abs(sf["r"][2]) < 1e-9
              and abs(vi[2]) < 1e-12 and abs(vf[2]) < 1e-12)
    if not planar:
        return np.nan, unsigned

    # Velocity-based signed angle: exact modulo 2 pi.
    theta_velocity = np.arctan2(vi[0] * vf[1] - vi[1] * vf[0], vi @ vf)
    # Winding estimate from the swept azimuth (straight-asymptote geometry, see module docstring).
    r_in, r_out = np.linalg.norm(si["r"]), np.linalg.norm(sf["r"])
    b_in, b_out = impact_parameter_of(y_initial), impact_parameter_of(y_final)
    # A free particle sweeps pi - asin(b_in/r_in) - asin(b_out/r_out) in the sense of L_z.
    sense = np.sign(si["r"][0] * vi[1] - si["r"][1] * vi[0])
    free_sweep = np.pi - np.arcsin(b_in / r_in) - np.arcsin(b_out / r_out)
    theta_swept = (sf["psi"][0] - si["psi"][0]) - sense * free_sweep
    windings = np.round((theta_swept - theta_velocity) / (2 * np.pi))
    return theta_velocity + 2 * np.pi * windings, unsigned
