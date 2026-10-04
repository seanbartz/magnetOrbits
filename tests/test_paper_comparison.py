"""Stage 5 — asymptotic in-state correction, and comparison with arXiv:2212.00876.

In-state correction checks
--------------------------
Without the incoming-leg correction, a moment direction specified at r0 differs from the
asymptotic one by O(1/(I E r0)), so r0-convergence of any observable is first order. With the
first-order correction it must become second order (measured 2.02, 1.98 on two generic cases;
3.0 head-to-tail, where the torque on the approach line is already O(b/r^4)).

Paper comparison (tests/published/bartz_shaw_2022.py)
------------------------------------------------------
Our converged collision edge for their configuration is b_c = 5.754590 (r0 and rtol converged to
~2e-8), versus their quoted 4.65: a 24% DISCREPANCY, recorded as a strict xfail. Checked and
ruled out on our side: r0 (1000 -> 4000 changes b_c by 2e-8), rtol (1e-10 -> 1e-12: < 1e-10),
alignment with r_hat at a finite start separation r_s (b_c = 5.66 even at r_s = 8R), and the
sticking-vs-touching definition (a first-bounce sticking estimate with their Gamma = 0.05 gives
b = 5.54). An energy 2.07x larger (E = 0.0694) would put our edge at 4.65.
"""

import numpy as np
import pytest

from physics import central
from physics.capture import find_capture_edge
from physics.solve import initial_state, run_trajectory
from tests.published import bartz_shaw_2022 as paper
from tests.validated import STAGE5_HEAD_TO_TAIL_EDGE

X_HAT = np.array([1.0, 0.0, 0.0])
OUR_EDGE = STAGE5_HEAD_TO_TAIL_EDGE   # converged value (see tests/validated.py)


def generic_case(index):
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(5):
        m1, m2 = rng.normal(size=3), rng.normal(size=3)
        cases.append((rng.uniform(0.01, 0.3), rng.uniform(2.0, 8.0), m1, m2))
    return cases[index]


def r0_convergence_order(energy, b, m1, m2, correction):
    values = []
    for r0 in (500.0, 1000.0, 2000.0):
        res = run_trajectory(energy, b, m1, m2, start_radius=r0, outgoing_radius=4000.0,
                             rtol=1e-12, atol=1e-14, asymptotic_correction=correction)
        values.append(np.concatenate([[res["deflection_3d"]], res["y_end"][12:18]]))
    diffs = np.linalg.norm(np.diff(np.array(values), axis=0), axis=1)
    return np.log2(diffs[0] / diffs[1])


@pytest.mark.slow
@pytest.mark.parametrize("index", [1, 3])
def test_correction_raises_r0_order_from_one_to_two(index):
    energy, b, m1, m2 = generic_case(index)
    assert r0_convergence_order(energy, b, m1, m2, correction=False) == pytest.approx(1.0, abs=0.15)
    assert r0_convergence_order(energy, b, m1, m2, correction=True) == pytest.approx(2.0, abs=0.15)


def test_correction_is_identity_when_torques_vanish():
    z = np.array([0.0, 0.0, 1.0])
    a = initial_state(0.03, 5.0, z, -z, asymptotic_correction=True)
    b = initial_state(0.03, 5.0, z, -z, asymptotic_correction=False)
    assert np.array_equal(a, b)


def test_correction_rejects_spinning_in_state():
    with pytest.raises(ValueError):
        initial_state(0.03, 5.0, X_HAT, X_HAT, w1=(0, 0, 0.1))


def test_energy_mapping():
    assert paper.ENERGY_TILDE == pytest.approx(0.033501, rel=1e-4)


@pytest.mark.slow
def test_our_edge_regression():
    found = find_capture_edge(paper.ENERGY_TILDE, X_HAT, X_HAT, 5.70, 5.80, rel_tol=1e-10)
    assert found["edge"] == pytest.approx(OUR_EDGE, abs=5e-7)
    # 0.025% below the locked head-to-tail law (spin included; 5.756012), its adiabatic limit,
    # and 7% above the -2/r^3 law with the spin ignored (5.372901).
    locked = central.locked_capture_impact_parameter(paper.ENERGY_TILDE)
    assert found["edge"] / locked == pytest.approx(0.99975, abs=2e-5)
    spin_free = central.capture_impact_parameter(paper.ENERGY_TILDE, 2.0)
    assert found["edge"] / spin_free == pytest.approx(1.0710, abs=1e-3)


@pytest.mark.xfail(strict=True, reason="documented 24% discrepancy with arXiv:2212.00876 Fig. 4 "
                                       "(see module docstring); strict so any change is noticed")
def test_paper_max_colliding_impact_parameter():
    assert OUR_EDGE == pytest.approx(paper.TARGET_MAX_IMPACT_PARAMETER, abs=paper.ABS_TOL), (
        f"ours {OUR_EDGE:.5f} vs theirs {paper.TARGET_MAX_IMPACT_PARAMETER} ({paper.CITATION}; {paper.SOURCE})")
