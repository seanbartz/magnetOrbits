"""The potential, force and torque: polar EOM vs. the vector formulas."""
import numpy as np
import pytest

from magdip import dynamics as dyn
from magdip.dimensional import DipolePair
from magdip.dynamics import IRHO, ITHETA, IPHI1, IPHI2, IPR, IL, IS1, IS2

RNG = np.random.default_rng(20240611)


def test_shape_factor_matches_vector_expression():
    """W = 3(m1.rhat)(m2.rhat) - m1.m2 with unit moments, for random angles."""
    a1, a2 = RNG.uniform(-np.pi, np.pi, (2, 500))
    m1 = np.stack([np.cos(a1), np.sin(a1)])
    m2 = np.stack([np.cos(a2), np.sin(a2)])
    rhat = np.array([1.0, 0.0])[:, None]        # alpha measured from the line of centres
    direct = 3.0 * (m1 * rhat).sum(0) * (m2 * rhat).sum(0) - (m1 * m2).sum(0)
    assert np.allclose(dyn.shape_factor(a1, a2), direct, atol=1e-14)


def test_locked_configuration_gives_inverse_cube():
    """Nose-to-tail is the global minimum, W = 2, U = -1/rho^3."""
    assert dyn.shape_factor(0.0, 0.0) == pytest.approx(2.0)
    assert dyn.shape_factor(np.pi, np.pi) == pytest.approx(2.0)
    assert dyn.potential(2.0, 0.0, 0.0) == pytest.approx(-1.0 / 8.0)
    a1, a2 = RNG.uniform(-np.pi, np.pi, (2, 2000))
    assert dyn.shape_factor(a1, a2).max() <= 2.0 + 1e-12


def test_shape_factor_gradient():
    a1, a2 = RNG.uniform(-np.pi, np.pi, (2, 200))
    h = 1e-6
    g1 = (dyn.shape_factor(a1 + h, a2) - dyn.shape_factor(a1 - h, a2)) / (2 * h)
    g2 = (dyn.shape_factor(a1, a2 + h) - dyn.shape_factor(a1, a2 - h)) / (2 * h)
    W1, W2 = dyn.shape_factor_grad(a1, a2)
    assert np.allclose(W1, g1, atol=1e-8)
    assert np.allclose(W2, g2, atol=1e-8)


def test_rhs_is_hamiltons_equations():
    """dq/dt = +dH/dp and dp/dt = -dH/dq, checked by central differences."""
    for _ in range(25):
        y = np.concatenate([[RNG.uniform(0.8, 5.0)], RNG.uniform(-np.pi, np.pi, 3),
                            RNG.uniform(-1.0, 1.0, 4)])
        f = dyn.rhs(0.0, y)
        h = 1e-7
        for q, p in ((IRHO, IPR), (ITHETA, IL), (IPHI1, IS1), (IPHI2, IS2)):
            yp = y.copy(); yp[p] += h
            ym = y.copy(); ym[p] -= h
            assert f[q] == pytest.approx((dyn.energy(yp) - dyn.energy(ym)) / (2 * h),
                                         rel=1e-6, abs=1e-9)
            yp = y.copy(); yp[q] += h
            ym = y.copy(); ym[q] -= h
            assert f[p] == pytest.approx(-(dyn.energy(yp) - dyn.energy(ym)) / (2 * h),
                                         rel=1e-5, abs=1e-8)


def test_polar_rhs_matches_cartesian_vector_formulas():
    """Independent check against F1 = (3mu0/4pi r^4)[...] and tau = m x B."""
    pair = DipolePair(mass=0.031, inertia=2.7e-6, moment=0.45)
    for _ in range(20):
        y = np.concatenate([[RNG.uniform(1.0, 6.0)], RNG.uniform(-np.pi, np.pi, 3),
                            RNG.uniform(-0.5, 0.5, 4)])
        state = pair.from_dimensionless(y)
        # dimensional derivatives, mapped back into dimensionless polar form
        a, t0 = pair.length_scale, pair.time_scale
        F1 = pair.force_on_1(state)
        tq1, tq2 = pair.field_torques(state)
        rx, ry = state[0] - state[2], state[1] - state[3]
        rmag = np.hypot(rx, ry)
        rhat = np.array([rx, ry]) / rmag
        that = np.array([-ry, rx]) / rmag
        # relative acceleration is 2*F1/m = F1/mu
        acc = 2.0 * F1 / pair.mass
        thdot = y[IL] / y[IRHO] ** 2 / t0
        # p_rho' = rhat.acc + rho thdot^2 ; L' = r * (that.acc)
        pr_dot = (rhat.dot(acc) + rmag * thdot ** 2) * t0 ** 2 / a
        L_dot = rmag * that.dot(acc) * t0 ** 2 / a ** 2
        s1_dot = tq1 / pair.inertia * t0 ** 2
        s2_dot = tq2 / pair.inertia * t0 ** 2

        f = dyn.rhs(0.0, y)
        assert f[IPR] == pytest.approx(pr_dot, rel=1e-11)
        assert f[IL] == pytest.approx(L_dot, rel=1e-11)
        assert f[IS1] == pytest.approx(s1_dot, rel=1e-11)
        assert f[IS2] == pytest.approx(s2_dot, rel=1e-11)


def test_potential_matches_dimensional():
    pair = DipolePair(mass=0.031, inertia=2.7e-6, moment=0.45)
    for _ in range(20):
        y = np.concatenate([[RNG.uniform(1.0, 6.0)], RNG.uniform(-np.pi, np.pi, 3),
                            RNG.uniform(-0.5, 0.5, 4)])
        state = pair.from_dimensionless(y)
        U = dyn.potential(y[IRHO], y[IPHI1] - y[ITHETA], y[IPHI2] - y[ITHETA])
        assert U * pair.energy_scale == pytest.approx(pair.potential(state), rel=1e-12)


def test_torques_sum_to_minus_orbital_torque():
    """J = L + s1 + s2 is conserved: the three torques must cancel."""
    y = np.concatenate([[2.3], RNG.uniform(-np.pi, np.pi, 3), RNG.uniform(-1, 1, 4)])
    f = dyn.rhs(0.0, y)
    assert f[IL] + f[IS1] + f[IS2] == pytest.approx(0.0, abs=1e-15)
