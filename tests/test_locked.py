"""Step 3: the locked-dipole (adiabatic) limit, analytically and numerically.

This is the validation case for the whole project -- the one regime where the answer
is known in closed form.
"""
import numpy as np
import pytest

from magdip import locked, scatter
from magdip.dynamics import IRHO


def test_barrier_position_and_height():
    """rho* = 3/L^2 and U_eff(rho*) = L^6/54, against a fine numerical maximum."""
    for L in (0.4, 0.9, 1.3, 2.0):
        rho = np.linspace(0.2 * 3 / L ** 2, 6 * 3 / L ** 2, 2_000_001)
        Ueff = 0.5 * (L / rho) ** 2 - 1.0 / rho ** 3
        i = np.argmax(Ueff)
        assert rho[i] == pytest.approx(locked.barrier_radius(L), rel=1e-5)
        assert Ueff[i] == pytest.approx(locked.barrier_height(L), rel=1e-8)


def test_bmax_formula_is_the_barrier_condition():
    """b_max is exactly where the barrier height equals E."""
    for E in (0.003, 0.05, 0.4, 3.0):
        b = locked.b_max(E)
        L = np.sqrt(2.0 * E) * b
        assert locked.barrier_height(L) == pytest.approx(E, rel=1e-12)
    assert locked.BMAX_COEFF == pytest.approx((27.0 / 4.0) ** (1.0 / 6.0))


def test_cross_section_exponent():
    """Planar sigma = 2 b_max ~ E^(-1/3) (the 3-D pi b_max^2 gives E^(-2/3))."""
    E = np.array([1e-3, 1e-2, 1e-1, 1.0])
    p = np.polyfit(np.log(E), np.log(locked.cross_section(E)), 1)[0]
    assert p == pytest.approx(-1.0 / 3.0, abs=1e-12)
    p3 = np.polyfit(np.log(E), np.log(np.pi * locked.b_max(E) ** 2), 1)[0]
    assert p3 == pytest.approx(-2.0 / 3.0, abs=1e-12)


@pytest.mark.parametrize("E", [0.01, 0.1, 1.0, 5.0])
def test_numerical_capture_boundary_matches_bmax(E):
    """Integrating the locked-dipole EOM reproduces the analytic b_max."""
    bm = locked.b_max(E)
    assert locked.run_trajectory(E, bm * 0.999)[0] == "collapse"
    assert locked.run_trajectory(E, bm * 1.001)[0] == "escape"


@pytest.mark.parametrize("E,b", [(0.1, 3.1), (0.1, 5.0), (0.1, 12.0),
                                 (1.0, 1.5), (1.0, 4.0), (0.01, 7.0)])
def test_quadrature_deflection_matches_integration(E, b):
    """The closed-form deflection integral vs. the locked-dipole ODE."""
    outcome, chi = locked.run_trajectory(E, b, rho0=4000.0)
    assert outcome == "escape"
    assert chi == pytest.approx(locked.deflection_angle(E, b), abs=2e-5)


def test_locked_limit_is_the_global_minimum_of_the_full_potential():
    """W = 2 is attained only nose-to-tail, which is why the limit is 1/rho^3."""
    from magdip import dynamics as dyn
    g = np.linspace(-np.pi, np.pi, 721)
    A1, A2 = np.meshgrid(g, g, indexing="ij")
    W = dyn.shape_factor(A1, A2)
    assert W.max() == pytest.approx(2.0, abs=1e-5)
    best = np.argwhere(W > 2.0 - 1e-6)
    for i, j in best:
        assert abs(np.sin(A1[i, j])) < 1e-3 and abs(np.sin(A2[i, j])) < 1e-3
        assert np.cos(A1[i, j]) * np.cos(A2[i, j]) > 0


@pytest.mark.slow
@pytest.mark.parametrize("E,tol", [(0.03, 0.20), (0.01, 0.12), (0.003, 0.06)])
def test_full_dynamics_approaches_locked_limit_as_E_goes_to_zero(E, tol):
    """Etil is the adiabaticity parameter: as Etil -> 0 the capture edge -> b_max.

    Started nose-to-tail (phi1 = phi2 = 0), i.e. in the basin of the locked
    configuration; other initial orientations need not follow the adiabatic branch.
    """
    bm = locked.b_max(E)
    bs = np.linspace(0.6 * bm, 1.6 * bm, 600)
    r = scatter.run_batch(E, bs, 0.0, 0.0, regularized=True, rtol=1e-11)
    assert r["collapsed"].any()
    edge = bs[np.max(np.where(r["collapsed"])[0])]
    assert abs(edge / bm - 1.0) < tol, (E, edge, bm, edge / bm)


@pytest.mark.slow
def test_capture_edge_converges_monotonically_toward_bmax():
    """The departure from the locked prediction shrinks as Etil decreases."""
    ratios = []
    for E in (0.03, 0.01, 0.003):
        bm = locked.b_max(E)
        bs = np.linspace(0.6 * bm, 1.6 * bm, 600)
        r = scatter.run_batch(E, bs, 0.0, 0.0, regularized=True, rtol=1e-11)
        ratios.append(bs[np.max(np.where(r["collapsed"])[0])] / bm)
    dev = [abs(x - 1.0) for x in ratios]
    assert dev[0] > dev[1] > dev[2], ratios
