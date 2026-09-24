"""Step 2: E and J drift are the primary correctness checks."""
import numpy as np
import pytest

from magdip import dynamics as dyn
from magdip import integrators as itg
from magdip import regularize as reg
from magdip import scatter
from magdip.dynamics import IRHO, IL

# (Etil, btil, phi1, phi2) spanning capture and fly-by, aligned and skewed
CASES = [
    (0.10, 5.0, 0.0, 0.0),
    (0.10, 2.0, 0.9, -2.1),
    (1.00, 1.5, 2.4, 0.7),
    (0.01, 9.0, -1.2, 1.2),
    (0.30, 0.3, 3.0, -0.6),
]


@pytest.mark.parametrize("E,b,p1,p2", CASES)
def test_scipy_reference_conserves_E_and_J(E, b, p1, p2):
    r = scatter.run_trajectory(E, b, p1, p2, rtol=1e-12, atol=1e-12)
    assert r["outcome_name"] in ("escape", "collapse")
    # energy is compared against the depth of the well actually visited, since |U|
    # grows like rho^-3 and absolute drift there says nothing about relative accuracy
    scale = max(abs(r["E0"]), 1.0 / r["rho_min"] ** 3)
    assert abs(r["dE"]) / scale < 1e-10
    assert abs(r["dJ"]) < 1e-11 * max(abs(r["J0"]), 1.0)


@pytest.mark.parametrize("E,b,p1,p2", CASES)
def test_batch_integrator_conserves_E_and_J(E, b, p1, p2):
    for regularized in (False, True):
        r = scatter.run_batch(E, b, p1, p2, rtol=1e-12, atol=1e-13,
                              regularized=regularized)
        rho_min = min(scatter.DEFAULT_RHO_CAPTURE, float(r["y_end"][IRHO]))
        scale = max(abs(float(r["E0"])), 1.0 / rho_min ** 3)
        assert abs(float(r["dE"])) / scale < 1e-9
        assert abs(float(r["dJ"])) < 1e-10 * max(abs(float(r["J0"])), 1.0)


def test_J_conserved_while_L_is_not():
    """The whole point: orbital angular momentum is exchanged with spin."""
    r = scatter.run_trajectory(0.3, 2.5, 1.1, -0.4, rtol=1e-12, atol=1e-12)
    L = r["sol"].y[IL]
    assert abs(r["dJ"]) < 1e-11
    assert L.max() - L.min() > 0.05 * abs(r["J0"]), "L should vary substantially"


def test_L_is_conserved_for_symmetric_nonrotating_configuration():
    """A sanity counter-check: with no torque, L must be exactly constant.

    alpha1 = alpha2 = pi/2 is a stationary point of W (side-by-side, parallel), so both
    field torques vanish and the force is purely central for as long as it is held.
    """
    from magdip.dynamics import shape_factor_grad
    W1, W2 = shape_factor_grad(np.pi / 2, np.pi / 2)
    assert W1 == pytest.approx(0.0, abs=1e-15)
    assert W2 == pytest.approx(0.0, abs=1e-15)
    y = np.array([3.0, 0.0, np.pi / 2, np.pi / 2, -0.4, 1.0, 0.0, 0.0])
    f = dyn.rhs(0.0, y)
    assert f[IL] == pytest.approx(0.0, abs=1e-15)


def test_energy_conserved_through_deep_collapse():
    """Regularized integration all the way down to rho = 1e-5."""
    y0 = scatter.initial_state(0.1, 0.5, 0.3, -0.8, rho0=50.0)
    E0 = dyn.energy(y0)
    f = reg.make_rhs(E0)
    ev = [itg.Event(lambda s, z: z[IRHO] - 1e-5, -1, 2)]
    res = itg.integrate_batch(f, reg.lift(y0)[:, None], np.inf, ev,
                              rtol=1e-12, atol=1e-13)
    assert res["outcome"][0] == 2
    y_end = reg.drop(res["y"])[:, 0]
    rho_end = y_end[IRHO]
    assert abs(dyn.energy(y_end) - E0) / (1.0 / rho_end ** 3) < 1e-10
    assert abs(dyn.total_angular_momentum(y_end)
               - dyn.total_angular_momentum(y0)) < 1e-10


def test_no_bound_orbits_in_the_locked_limit():
    """U_eff = L^2/2rho^2 - 1/rho^3 has a maximum and no minimum.

    Recorded as a test because it shapes the whole project: the only asymptotic
    outcomes are escape and collapse -- there are no permanently trapped orbits to
    integrate for long times, so "long-lived complex" always means a transient.
    """
    L = 0.9
    rho = np.linspace(0.05, 60.0, 200_000)
    Ueff = 0.5 * (L / rho) ** 2 - 1.0 / rho ** 3
    i = np.argmax(Ueff)
    assert rho[i] == pytest.approx(3.0 / L ** 2, rel=1e-3)
    assert Ueff[i] == pytest.approx(L ** 6 / 54.0, rel=1e-3)
    # strictly decreasing beyond the barrier: no well, hence no stable orbit
    assert np.all(np.diff(Ueff[i:]) < 0)
