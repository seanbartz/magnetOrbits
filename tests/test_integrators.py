"""Integrator correctness: order, symplecticity, and agreement with SciPy."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from magdip import dynamics as dyn
from magdip import integrators as itg
from magdip import scatter
from magdip.dynamics import IRHO

Y0 = np.array([4.0, 0.3, 0.8, -1.4, -0.35, 1.1, 0.05, -0.02])


def _flow(y0, T, h, stages):
    n = int(round(T / h))
    y, _ = itg.integrate_gl(dyn.rhs, y0, h=T / n, nsteps=n, stages=stages)
    return y


@pytest.mark.parametrize("stages,order,hs", [(2, 4, (0.16, 0.08, 0.04)),
                                            (3, 6, (0.80, 0.40, 0.20))])
def test_gauss_legendre_order(stages, order, hs):
    """GL with s stages is order 2s.

    The step sizes differ per scheme only because GL3 reaches double-precision
    roundoff (~1e-14 here) by h = 0.16, where the measured rate would be meaningless.
    """
    T = 2.0
    ref = solve_ivp(dyn.rhs, (0.0, T), Y0, method="DOP853", rtol=1e-13,
                    atol=1e-16).y[:, -1]
    errs = np.array([np.max(np.abs(_flow(Y0, T, h, stages) - ref)) for h in hs])
    assert errs[-1] > 1e-15, "step sizes too small to measure the order"
    rates = np.log2(errs[:-1] / errs[1:])
    assert np.all(rates > order - 0.8), (hs, errs, rates)


def _symplectic_residual(step, h, eps=1e-6):
    """max |M^T J M - J| for the one-step map, M by central differences."""
    n = dyn.NDIM
    M = np.empty((n, n))
    for j in range(n):
        yp, ym = Y0.copy(), Y0.copy()
        yp[j] += eps
        ym[j] -= eps
        M[:, j] = (step(yp, h) - step(ym, h)) / (2 * eps)
    half = n // 2
    J = np.zeros((n, n))
    J[:half, half:] = np.eye(half)
    J[half:, :half] = -np.eye(half)
    return np.max(np.abs(M.T @ J @ M - J))


def _gl_step(stages):
    return lambda y, h: itg.gauss_legendre_step(dyn.rhs, 0.0, y, h, stages=stages)


def _dp_step(y, h):
    return itg._dp_stages(dyn.rhs, 0.0, y, h)[1]


@pytest.mark.parametrize("stages", [2, 3])
def test_gauss_legendre_step_is_symplectic(stages):
    """M^T J M = J -- the defining property -- and for *any* step size.

    The residual floor is finite-difference noise (~2e-10), not the scheme, so the
    signature of a genuinely symplectic method is that it stays flat as h grows while
    an explicit method's grows steeply (see the companion test below).
    """
    res = [_symplectic_residual(_gl_step(stages), h) for h in (0.4, 1.6, 3.2)]
    assert max(res) < 1e-8, res
    assert max(res) < 20 * min(res), ("residual should not grow with h", res)


def test_explicit_rk_step_is_not_symplectic():
    """The same check must fail for Dormand-Prince, or it has no teeth."""
    assert _symplectic_residual(_dp_step, 1.6) > 1e-7
    assert _symplectic_residual(_gl_step(2), 1.6) < 1e-8


def test_dense_output_is_high_order():
    """The event locator relies on the free DOPRI5 interpolant."""
    f = lambda t, y: np.array([y[1], -y[0]])
    y0 = np.array([1.0, 0.0])
    errs = []
    hs = [0.4, 0.2, 0.1]
    for h in hs:
        k, y1, _ = itg._dp_stages(f, 0.0, y0, h)
        e = max(abs(itg._dp_interp(y0, y1, h, k, th)[0] - np.cos(th * h))
                for th in np.linspace(0.05, 0.95, 19))
        errs.append(e)
    rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(rates > 4.5), (errs, rates)


def test_batch_matches_scipy_reference():
    """The vectorised driver must reproduce SciPy trajectory by trajectory."""
    E = 0.1
    bs = np.array([0.7, 2.5, 3.4, 4.5, 6.0, 9.0])
    batch = scatter.run_batch(E, bs, 0.4, -1.1, rtol=1e-12, atol=1e-13)
    for i, b in enumerate(bs):
        ref = scatter.run_trajectory(E, b, 0.4, -1.1, rtol=1e-12, atol=1e-12)
        assert scatter.OUTCOME_NAMES[batch["outcome"][i]] == ref["outcome_name"], b
        if ref["outcome_name"] == "escape":
            assert batch["chi"][i] == pytest.approx(ref["chi"], abs=1e-9)
            assert batch["t_end"][i] == pytest.approx(ref["t_end"], rel=1e-8)


def test_event_location_is_accurate():
    """Terminal states must sit on the event surface, not merely past it."""
    bs = np.linspace(0.5, 8.0, 60)
    r = scatter.run_batch(0.1, bs, 0.0, 0.0, rtol=1e-12, atol=1e-13)
    rho_end = r["y_end"][IRHO]
    col, esc = r["collapsed"], r["escaped"]
    assert np.allclose(rho_end[col], scatter.DEFAULT_RHO_CAPTURE, rtol=1e-10)
    assert np.allclose(rho_end[esc], scatter.DEFAULT_RHO0, rtol=1e-8)


def test_batch_and_regularized_agree():
    bs = np.linspace(0.3, 8.0, 120)
    a = scatter.run_batch(0.2, bs, 1.0, -0.5, rtol=1e-12, atol=1e-13)
    b = scatter.run_batch(0.2, bs, 1.0, -0.5, rtol=1e-12, atol=1e-13, regularized=True)
    assert np.array_equal(a["outcome"], b["outcome"])
    e = a["escaped"]
    assert np.nanmax(np.abs(a["chi"][e] - b["chi"][e])) < 1e-7
