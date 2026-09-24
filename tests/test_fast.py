"""Stage 7 — compiled integrator (physics/fast.py) against the scipy reference path.

The compiled path implements the SAME algorithm as scipy's DOP853 + solve_ivp events, so the
checks are stronger than "agrees to a tolerance":
* compiled RHS = equations_of_motion to roundoff (random states);
* Gauss-Legendre in-state quadrature = quad_vec to ~1e-14 (measured 1.3e-15);
* identical number of RHS evaluations (same accepted/rejected step sequence) and end states
  equal to roundoff, for generic 3D in-states in every stop mode and outcome;
* every reference-path known result re-run with method="numba": central-force quadrature
  deflection, analytic capture edges (both regimes), grazing contact via periapsis,
  stage-4 regression reference, stage-5 paper edge, stage-6b map points;
* the recomputed 32 x 32 production map matches tests/reference/map_torus32.npz
  (analysis/compare_maps.py; not a unit test because it takes ~1 min on 18 workers).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from physics import central, fast
from physics.capture import find_capture_edge
from physics.edges import sweep_point
from physics.solve import equations_of_motion, incoming_leg_correction, run_trajectory
from tests.published.bartz_shaw_2022 import ENERGY_TILDE

Z = np.array([0.0, 0.0, 1.0])
X_HAT = np.array([1.0, 0.0, 0.0])
REFERENCE = Path(__file__).parent / "reference"


def test_rhs_matches_reference():
    rng = np.random.default_rng(1)
    for _ in range(200):
        y = rng.normal(size=19)
        y[0:3] *= 3.0
        out = np.empty(19)
        fast.rhs(y, out, 0.8)
        ref = equations_of_motion(0.0, y, 0.8)
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-14)


def test_gauss_quadrature_matches_quad_vec():
    rng = np.random.default_rng(3)
    for _ in range(40):
        m1, m2 = rng.normal(size=3), rng.normal(size=3)
        m1, m2 = m1 / np.linalg.norm(m1), m2 / np.linalg.norm(m2)
        energy, b = rng.uniform(0.005, 1.0), rng.uniform(0.0, 12.0)
        r0 = rng.choice([250.0, 1000.0, 64000.0])
        x0 = -np.sqrt(r0**2 - b**2)
        a = np.concatenate(incoming_leg_correction(energy, b, x0, m1, m2, 0.8))
        g = np.concatenate(incoming_leg_correction(energy, b, x0, m1, m2, 0.8, quadrature="gauss"))
        assert np.max(np.abs(a - g)) <= 1e-13 * np.max(np.abs(a))


def generic_cases(n=6, seed=11):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0.01, 0.3), rng.uniform(1.0, 8.0), rng.normal(size=3), rng.normal(size=3))
            for _ in range(n)]


@pytest.mark.parametrize("stop", ["escape", "r_out"])
@pytest.mark.parametrize("correction", [True, False])
@pytest.mark.parametrize("case", range(6))
def test_same_steps_and_end_state(case, correction, stop):
    energy, b, m1, m2 = generic_cases()[case]
    kw = dict(stop=stop, asymptotic_correction=correction)
    ref = run_trajectory(energy, b, m1, m2, **kw)
    new = run_trajectory(energy, b, m1, m2, method="numba", **kw)
    assert new["outcome"] == ref["outcome"]
    assert new["contact_via"] == ref["contact_via"]
    assert new["n_rhs"] == ref["n_rhs"]
    assert new["t_end"] == pytest.approx(ref["t_end"], rel=1e-12)
    np.testing.assert_allclose(new["y_end"], ref["y_end"], rtol=1e-9, atol=1e-9)
    # Drift diagnostics agree to 0.1%, except at the roundoff floor (< 1e-12), where the
    # evaluation order legitimately changes the digits.
    for key in ("energy_drift", "angular_momentum_drift", "moment_norm_error"):
        assert new[key] == pytest.approx(ref[key], rel=1e-3, abs=1e-12)
    if np.isfinite(ref["deflection_3d"]):
        assert new["deflection_3d"] == pytest.approx(ref["deflection_3d"], abs=1e-10)


def test_undecided_and_timeout_outcomes():
    assert run_trajectory(0.0335, 6.0, X_HAT, X_HAT, outgoing_radius=4.5, method="numba")["outcome"] == "undecided"
    assert run_trajectory(0.0335, 6.0, X_HAT, X_HAT, max_time=100.0, method="numba")["outcome"] == "timeout"


@pytest.mark.parametrize("energy,b", [(0.01, 8.0), (0.03, 5.0), (0.03, 4.6), (0.1, 3.1), (0.3, 2.45), (1.0, 2.3)])
def test_central_deflection(energy, b):
    res = run_trajectory(energy, b, Z, -Z, method="numba")
    assert res["deflection"] == pytest.approx(central.deflection_angle(energy, b, 1.0), abs=1e-8)


@pytest.mark.parametrize("energy", [0.003, 0.03, 0.065, 0.3, 1.0])
def test_capture_edge_analytic(energy):
    expected = central.capture_impact_parameter(energy, 1.0)
    found = find_capture_edge(energy, Z, -Z, 0.9 * expected, 1.1 * expected, rel_tol=1e-9, method="numba")
    assert found["edge"] == pytest.approx(expected, rel=1e-9)


def test_grazing_contact_via_periapsis():
    edge = central.capture_impact_parameter(0.3, 1.0)
    res = run_trajectory(0.3, edge * (1 - 1e-7), Z, -Z, stop="escape", method="numba")
    assert res["outcome"] == "contact" and res["contact_via"] == "periapsis"


def test_stage4_regression_reference():
    ref = np.load(REFERENCE / "stage4_generic.npz")
    for k in range(len(ref["energy"])):
        res = run_trajectory(ref["energy"][k], ref["impact_parameter"][k], ref["m1"][k], ref["m2"][k],
                             asymptotic_correction=False, method="numba")
        assert res["outcome"] == ref["outcome"][k]
        np.testing.assert_allclose(res["y_end"][:18], ref["y_end"][k][:18], rtol=1e-7, atol=1e-7)


def test_paper_edge():
    found = find_capture_edge(ENERGY_TILDE, X_HAT, X_HAT, 5.70, 5.80, rel_tol=1e-10, method="numba")
    assert found["edge"] == pytest.approx(5.7545902, abs=5e-7)


def test_map_reference_points():
    ref = np.load(REFERENCE / "map_torus32.npz")
    settings = {**json.loads(str(ref["settings"])), "method": "numba"}
    n = len(ref["phi"])
    for i, j in [(0, 15), (1, 1), (0, 16), (7, 23)]:
        rec = sweep_point(settings["energy"], i, j, n, settings)
        assert rec["n_intervals"] == ref["n_intervals"][i, j]
        assert rec["outer_edge"] == pytest.approx(ref["outer_edge"][i, j], rel=2e-6)
        assert rec["capture_measure"] == pytest.approx(ref["capture_measure"][i, j], rel=2e-6)


def test_speedup():
    """Report-style check: compiled path at least 50x faster per trajectory (measured ~300x)."""
    import time
    energy, b, m1, m2 = generic_cases()[0]
    run_trajectory(energy, b, m1, m2, method="numba")          # compile / warm cache
    t0 = time.perf_counter()
    for _ in range(3):
        run_trajectory(energy, b, m1, m2)
    t_ref = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for _ in range(30):
        run_trajectory(energy, b, m1, m2, method="numba")
    t_fast = (time.perf_counter() - t0) / 30
    print(f"reference {t_ref * 1e3:.1f} ms, numba {t_fast * 1e3:.2f} ms, speedup {t_ref / t_fast:.0f}x")
    assert t_ref / t_fast > 50


def test_compare_maps_nan_and_exit_status(tmp_path):
    """analysis/compare_maps.py: a nan in only one map is a failure, and failure exits nonzero."""
    import subprocess
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
    from compare_maps import max_rel_diff
    a = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert max_rel_diff(a, a) == 0.0
    assert max_rel_diff(a, np.array([[1.0, 5.0], [2.0, 3.0]])) == np.inf
    zeros = np.array([[0.0, 1.0]])
    assert max_rel_diff(zeros, zeros) == 0.0                       # 0 == 0 is not a failure
    assert max_rel_diff(np.array([[1e-3, 1.0]]), zeros) == np.inf  # nonzero vs zero is
    # A deliberately different map must make the script exit with status 1.
    ref = np.load(REFERENCE / "map_torus32.npz")
    n = len(ref["phi"])
    (tmp_path / "meta.json").write_text(json.dumps({"n_phi": n, "settings": json.loads(str(ref["settings"])),
                                                    "fundamental": False, "git_hash": "x", "git_dirty": False}))
    with open(tmp_path / "points.jsonl", "w") as f:
        for i in range(n):
            for j in range(n):
                f.write(json.dumps({"i": i, "j": j, "outer_edge": float(ref["outer_edge"][i, j]) * 1.01,
                                    "capture_measure": float(ref["capture_measure"][i, j]),
                                    "n_intervals": int(ref["n_intervals"][i, j]), "n_timeouts": 0,
                                    "seconds": 0.0}) + "\n")
    script = Path(__file__).resolve().parents[1] / "analysis" / "compare_maps.py"
    out = subprocess.run([sys.executable, str(script), str(tmp_path), str(REFERENCE / "map_torus32.npz")],
                         capture_output=True, text=True)
    assert out.returncode == 1 and "FAIL" in out.stdout
