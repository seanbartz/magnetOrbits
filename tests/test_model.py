"""Stage 1 — the interaction model (physics/model.py).

Checks
------
* F2 = -grad_r U by central finite differences at random configurations.
* tau_i = -dU/d(theta_i) by actually rotating m_i through +-eps about each axis.
* tau_1 + tau_2 + r x F2 = 0 (total angular momentum conservation) to machine precision.
* In-plane moments reproduce the paper's closed form (arXiv:2212.00876, Eq. 6):
      U = -(1/2 r^3) [ 3 cos(2 theta - phi1 - phi2) + cos(phi1 - phi2) ]   (units of U0).
* Special configurations: +z/-z gives U = -1/r^3 with zero torque *exactly*;
  head-to-tail gives U = -2/r^3 and F2 = -6/r^4 r_hat.
* Radial-force lower bound -6/r^4 holds over random orientations.
"""

import numpy as np
import pytest

from physics.model import MIN_RADIAL_FORCE_COEFF, force_on_2, potential, torques

RNG = np.random.default_rng(20260923)


def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def random_config(rng):
    """Random separation 2..6 in a random direction, random moment directions."""
    r_vec = rng.uniform(2.0, 6.0) * random_unit(rng)
    return r_vec, random_unit(rng), random_unit(rng)


def rotate(vec, axis, angle):
    """Rodrigues rotation of vec about unit axis by angle."""
    return (vec * np.cos(angle) + np.cross(axis, vec) * np.sin(angle)
            + axis * (axis @ vec) * (1 - np.cos(angle)))


@pytest.mark.parametrize("trial", range(20))
def test_force_is_minus_gradient(trial):
    r_vec, m1, m2 = random_config(RNG)
    step = 1e-6
    grad = np.array([
        (potential(r_vec + step * e, m1, m2) - potential(r_vec - step * e, m1, m2)) / (2 * step)
        for e in np.eye(3)])
    force = force_on_2(r_vec, m1, m2)
    # Central differences: truncation O(step^2 U''') ~ 1e-12, roundoff ~ 1e-16/1e-6 ~ 1e-10.
    assert np.allclose(force, -grad, rtol=0, atol=1e-8 * np.max(np.abs(force)) + 1e-12)


@pytest.mark.parametrize("trial", range(20))
def test_torque_is_minus_angular_derivative(trial):
    r_vec, m1, m2 = random_config(RNG)
    tau1, tau2 = torques(r_vec, m1, m2)
    eps = 1e-6
    for axis in np.eye(3):
        dU1 = (potential(r_vec, rotate(m1, axis, eps), m2)
               - potential(r_vec, rotate(m1, axis, -eps), m2)) / (2 * eps)
        dU2 = (potential(r_vec, m1, rotate(m2, axis, eps))
               - potential(r_vec, m1, rotate(m2, axis, -eps))) / (2 * eps)
        assert tau1 @ axis == pytest.approx(-dU1, abs=1e-9)
        assert tau2 @ axis == pytest.approx(-dU2, abs=1e-9)


@pytest.mark.parametrize("trial", range(50))
def test_total_torque_vanishes(trial):
    r_vec, m1, m2 = random_config(RNG)
    tau1, tau2 = torques(r_vec, m1, m2)
    total = tau1 + tau2 + np.cross(r_vec, force_on_2(r_vec, m1, m2))
    scale = np.linalg.norm(tau1) + np.linalg.norm(tau2)
    assert np.linalg.norm(total) < 1e-14 * max(scale, 1.0)


@pytest.mark.parametrize("trial", range(20))
def test_in_plane_matches_paper_closed_form(trial):
    theta, phi1, phi2 = RNG.uniform(0, 2 * np.pi, size=3)
    r = RNG.uniform(2.0, 6.0)
    r_vec = r * np.array([np.cos(theta), np.sin(theta), 0.0])
    m1 = np.array([np.cos(phi1), np.sin(phi1), 0.0])
    m2 = np.array([np.cos(phi2), np.sin(phi2), 0.0])
    paper = -(3 * np.cos(2 * theta - phi1 - phi2) + np.cos(phi1 - phi2)) / (2 * r**3)
    assert potential(r_vec, m1, m2) == pytest.approx(paper, rel=1e-13)


def test_antiparallel_z_is_central_minus_one_over_r_cubed():
    z = np.array([0.0, 0.0, 1.0])
    for trial in range(20):
        ang = RNG.uniform(0, 2 * np.pi)
        r = RNG.uniform(2.0, 50.0)
        r_vec = r * np.array([np.cos(ang), np.sin(ang), 0.0])
        assert potential(r_vec, z, -z) == pytest.approx(-1 / r**3, rel=1e-14)
        assert np.allclose(force_on_2(r_vec, z, -z), -3 / r**4 * r_vec / r, rtol=1e-14, atol=0)
        tau1, tau2 = torques(r_vec, z, -z)
        # Exactly zero in floating point: this is what makes +z/-z an invariant manifold numerically.
        assert np.all(tau1 == 0.0) and np.all(tau2 == 0.0)


def test_head_to_tail():
    r_hat = random_unit(RNG)
    r = 3.7
    r_vec = r * r_hat
    assert potential(r_vec, r_hat, r_hat) == pytest.approx(-2 / r**3, rel=1e-14)
    assert np.allclose(force_on_2(r_vec, r_hat, r_hat), -6 / r**4 * r_hat, rtol=1e-13, atol=0)


def test_radial_force_lower_bound():
    worst = np.inf
    for trial in range(20000):
        r_vec, m1, m2 = random_config(RNG)
        r = np.linalg.norm(r_vec)
        fr = force_on_2(r_vec, m1, m2) @ (r_vec / r) * r**4
        worst = min(worst, fr)
    assert worst >= MIN_RADIAL_FORCE_COEFF - 1e-12
    # Random sampling should come close to the bound (attained head-to-tail).
    assert worst < MIN_RADIAL_FORCE_COEFF + 0.5
