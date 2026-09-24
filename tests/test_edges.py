"""Stage 6b — capture profiles over b and the orientation-map machinery.

Checks
------
* +z/-z (exact central case): one contact interval [b_min, b_c], b_c analytic.
* Head-to-tail (0, 0) at the paper energy: outer edge = stage-5 value 5.7545902.
* Symmetries of the in-plane map, evaluated directly on a generic orientation:
      (phi1, phi2) ~ (phi1 + pi, phi2 + pi)   (global flip)
      (phi1, phi2) ~ (phi2, phi1)             (exchange composed with the flip)
* Scan refinement: the interval structure is unchanged when n_scan is quadrupled.
* Bookkeeping: fundamental domain covers every orbit exactly once; load_map expands a
  fundamental-domain run onto the full torus; sweep_point is the same computation as
  in_plane_profile (one code path).
The full-torus symmetry check on the 32 x 32 production map is in analysis/map_checks.py.
"""

import json

import numpy as np
import pytest

from physics import central
from physics.edges import (capture_profile, fundamental_domain, in_plane_profile, load_map,
                           sweep_point, symmetry_orbit)
from tests.published.bartz_shaw_2022 import ENERGY_TILDE

Z = np.array([0.0, 0.0, 1.0])
SETTINGS = dict(b_min=0.05, b_max=10.0, n_scan=200, rel_tol=1e-6)
GENERIC = (0.7, -2.1)


@pytest.mark.slow
def test_antiparallel_z_single_interval():
    prof = capture_profile(ENERGY_TILDE, Z, -Z, **SETTINGS)
    b_c = central.capture_impact_parameter(ENERGY_TILDE, 1.0)
    assert prof["n_intervals"] == 1
    lo, hi = prof["intervals"][0]
    assert lo == SETTINGS["b_min"]
    assert hi == pytest.approx(b_c, rel=2e-6)


@pytest.mark.slow
def test_head_to_tail_matches_stage5():
    prof = in_plane_profile(ENERGY_TILDE, 0.0, 0.0, **SETTINGS)
    assert prof["n_intervals"] == 1
    assert prof["outer_edge"] == pytest.approx(5.7545902, abs=1e-5)


@pytest.mark.slow
def test_in_plane_symmetries():
    phi1, phi2 = GENERIC
    base = in_plane_profile(ENERGY_TILDE, phi1, phi2, **SETTINGS)
    flipped = in_plane_profile(ENERGY_TILDE, phi1 + np.pi, phi2 + np.pi, **SETTINGS)
    swapped = in_plane_profile(ENERGY_TILDE, phi2, phi1, **SETTINGS)
    for other in (flipped, swapped):
        assert other["n_intervals"] == base["n_intervals"]
        np.testing.assert_allclose(other["edges"], base["edges"], rtol=2e-6)


@pytest.mark.slow
def test_scan_refinement_keeps_structure():
    coarse = in_plane_profile(ENERGY_TILDE, *GENERIC, **SETTINGS)
    fine = in_plane_profile(ENERGY_TILDE, *GENERIC, **{**SETTINGS, "n_scan": 800})
    assert fine["n_intervals"] == coarse["n_intervals"]
    np.testing.assert_allclose(fine["edges"], coarse["edges"], rtol=2e-6)


@pytest.mark.parametrize("n", [4, 8, 32])
def test_fundamental_domain_covers_each_orbit_once(n):
    reps = fundamental_domain(n)
    covered = [p for r in reps for p in symmetry_orbit(*r, n)]
    assert sorted(covered) == sorted((i, j) for i in range(n) for j in range(n))
    assert len(covered) == len(set(covered))


def test_load_map_expands_symmetry(tmp_path):
    n = 8
    (tmp_path / "meta.json").write_text(json.dumps({"n_phi": n, "settings": {}}))
    with open(tmp_path / "points.jsonl", "w") as f:
        for i, j in fundamental_domain(n):
            f.write(json.dumps({"i": i, "j": j, "outer_edge": 10.0 * i + j, "capture_measure": 1.0,
                                "n_intervals": 1, "n_timeouts": 0}) + "\n")
    data = load_map(tmp_path)
    assert not np.any(np.isnan(data["outer_edge"]))
    for i, j in fundamental_domain(n):
        for a, b in symmetry_orbit(i, j, n):
            assert data["outer_edge"][a, b] == 10.0 * i + j


@pytest.mark.slow
def test_sweep_point_is_in_plane_profile():
    n = 8
    rec = sweep_point(ENERGY_TILDE, 1, 6, n, SETTINGS)
    direct = in_plane_profile(ENERGY_TILDE, 2 * np.pi / n, 2 * np.pi * 6 / n, **SETTINGS)
    assert rec["edges"] == [float(e) for e in direct["edges"]]
    assert rec["scan_codes"] == direct["scan_codes"]


@pytest.mark.slow
def test_regression_against_production_map():
    """Recompute two points of the stored 32 x 32 production map (tests/reference/map_torus32.npz):
    one two-interval point on the phi1 + phi2 = 15 pi/16 diagonal and one near head-to-tail."""
    from pathlib import Path
    ref = np.load(Path(__file__).parent / "reference" / "map_torus32.npz")
    settings = json.loads(str(ref["settings"]))
    n = len(ref["phi"])
    for i, j in [(0, 15), (1, 1)]:
        rec = sweep_point(settings["energy"], i, j, n, settings)
        assert rec["n_intervals"] == ref["n_intervals"][i, j]
        assert rec["outer_edge"] == pytest.approx(ref["outer_edge"][i, j], rel=1e-9)
        assert rec["capture_measure"] == pytest.approx(ref["capture_measure"][i, j], rel=1e-9)
