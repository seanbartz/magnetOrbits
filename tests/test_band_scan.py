"""Stage 9 — the fractal escape-window band vs energy (analysis/band_scan.py).

Checks
------
* grids: both across lattices contain the 32 x 32 torus offsets (d = 0, -pi/16), the wide one
  contains the stage-8c lattice at a fixed index offset, and every other along row is a torus row;
* escape windows are read from edge lists ('ce' followed by 'ec'), not from interval counts;
* row measures: the sharp edge is the largest b_out jump, the window range spans n_intervals >= 2;
* band points: the most-edges and median-d window points on the exponent rows, with intervals
  that contain their windows; survey points spread over each row's window d range;
* the numerics-consistency rule accepts Poisson-scale differences and rejects systematic ones;
* a stored exponent entry is reused only for exactly the same point, settings and sample size;
* torus_index recognizes grid angles (mod 2 pi) and rejects off-grid ones;
* regression: 4x-scan strip points recomputed at two energies equal tests/reference/band_maps.npz
  (same interval count; b_out to B_OUT_TOLERANCE, the bisection-limited precision).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from band_scan import (B_OUT_TOLERANCE, EXPONENT_ROWS, NARROW, WIDE, band_measures, band_points,   # noqa: E402
                       escape_windows, matches, numerics_consistent, row_measures, strip_grid, survey_points,
                       torus_index)
from physics.edges import strip_axes, strip_point                                                 # noqa: E402

REFERENCE = ROOT / "tests" / "reference" / "band_maps.npz"


@pytest.mark.parametrize("across,step", [(NARROW, 1 / 2048), (WIDE, 1 / 512)])
def test_across_lattices_contain_torus_offsets(across, step):
    grid = {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, 32], "across_pi": across}
    along, d = strip_axes(grid)
    assert np.allclose(np.diff(d), np.pi * step)
    for target in (0.0, -np.pi / 16):
        assert np.min(np.abs(d - target)) < 1e-12
    assert np.allclose(along[::2], np.pi / 16 * np.arange(16))


def test_wide_lattice_contains_stage8_strip():
    _, d = strip_axes(strip_grid(0.1))
    _, d8 = strip_axes({"along_pi": [0.0, 1.0, 64], "across_pi": [-0.1875, 0.09375, 145]})
    assert np.allclose(d[96:96 + len(d8)], d8, atol=1e-14)


def test_escape_windows_from_edge_lists():
    rec = {"edges": [1.0, 1.2, 2.0, 2.1, 5.0], "edge_kinds": ["ce", "ec", "ce", "ec", "ce"]}
    assert escape_windows(rec) == [(1.0, 1.2), (2.0, 2.1)]
    assert escape_windows({"edges": [4.0], "edge_kinds": ["ce"]}) == []


def test_row_measures():
    across = np.linspace(-0.1, 0.0, 11)
    b_out = np.array([5.0, 4.9, 4.8, 3.6, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0])
    n_int = np.array([1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1])
    m = row_measures(b_out, n_int, across)
    assert m["sharp_index"] == 2 and m["sharp_edge"] == pytest.approx(-0.075)   # 4.8 -> 3.6
    assert (m["window_lo"], m["window_hi"], m["n_windows"]) == (pytest.approx(-0.06), pytest.approx(-0.03), 3)


def synthetic_strip():
    grid = {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, 32], "across_pi": [-0.1, 0.0, 11]}
    along, across = strip_axes(grid)
    n_int = np.ones((32, 11), dtype=int)
    records = {}
    for i in range(32):
        for j in range(11):
            edges, kinds = [5.0], ["ce"]
            if j in (4, 5, 7):
                n_int[i, j] = 2
                edges, kinds = [1.0, 1.1 + 0.01 * j, 5.0], ["ce", "ec", "ce"]
                if j == 5:
                    edges, kinds = [1.0, 1.05, 1.2, 1.3, 5.0], ["ce", "ec", "ce", "ec", "ce"]
            records[(i, j)] = {"edges": edges, "edge_kinds": kinds, "phi1": float(along[i]),
                               "phi2": float(np.pi - along[i] + across[j])}
    return {"grid": grid, "along": along, "across": across, "records": records, "n_intervals": n_int,
            "outer_edge": np.full((32, 11), 5.0)}


def test_band_points_choose_most_edges_and_median():
    pts = band_points(synthetic_strip())
    assert {p["i"] for p in pts.values()} == set(EXPONENT_ROWS)
    for p in pts.values():
        assert p["j"] == 5                        # most edges (5) and the median of [4, 5, 7] coincide
        assert p["b_lo"] < 1.0 and p["b_hi"] > 1.3
    assert len(pts) == len(EXPONENT_ROWS)


def test_survey_points_spread_over_the_window_range():
    pts = survey_points(synthetic_strip())
    for i in EXPONENT_ROWS:
        row = sorted(p["j"] for p in pts.values() if p["i"] == i)
        assert row == [4, 5, 7]                   # all three window points (fewer than SURVEY_PER_ROW)
    assert all(p["kind"] == "survey" and p["b_lo"] < 1.0 for p in pts.values())


def test_numerics_consistency_rule():
    ok = {"variants": {"production": {"uncertain": [58, 69, 55]}, "tight": {"uncertain": [61, 56, 59]}}}
    bad = {"variants": {"production": {"uncertain": [58, 69, 55]}, "tight": {"uncertain": [120, 140, 110]}}}
    assert numerics_consistent(ok) and not numerics_consistent(bad)


def test_band_points_skip_records_without_a_window_pair():
    data = synthetic_strip()
    for i in EXPONENT_ROWS:                       # two intervals, but a missed edge: no 'ce' -> 'ec' pair
        data["records"][(i, 4)]["edge_kinds"] = ["ce", "ce", "ce"]
    pts = band_points(data)
    assert all(p["j"] != 4 for p in pts.values())


def test_stored_entries_reused_only_for_the_same_point():
    point = {"kind": "band", "b_lo": 1.0, "b_hi": 1.3, "phi1": 0.5, "phi2": 2.1}
    settings = {"start_radius": 64000.0, "method": "numba"}
    stored = {**point, "settings": settings, "samples": 20000, "alpha": 0.3}
    assert matches(stored, point, settings, 20000)
    assert matches(stored, {**point, "b_hi": np.nextafter(1.3, 2.0)}, settings, 20000)   # last-bit bisection noise
    assert not matches(stored, {**point, "b_hi": 1.31}, settings, 20000)
    assert not matches(stored, point, {**settings, "start_radius": 128000.0}, 20000)
    assert not matches(stored, point, settings, 50000)


def test_band_measures_area_and_b_range():
    m = band_measures(synthetic_strip())
    ds, dd = np.pi / 32, np.pi * 0.01
    assert m["torus_fraction"] == pytest.approx(2 * 32 * 3 * ds * dd / (4 * np.pi ** 2))
    assert m["window_b_range"] == [1.0, pytest.approx(1.3)]
    assert m["rows_with_windows"] == 32 and not m["windows_at_range_end"]


def test_torus_index():
    assert torus_index(2 * np.pi * 3 / 32) == 3
    assert torus_index(-2 * np.pi / 32) == 31
    assert torus_index(2 * np.pi + 1e-13) == 0
    assert torus_index(0.01) is None


@pytest.mark.skipif(not REFERENCE.exists(), reason="band_maps.npz not stored yet")
@pytest.mark.parametrize("energy_key,i,j", [("E0.0335008", 16, 150), ("E0.005", 8, 100)])
def test_regression_against_stored_band_maps(energy_key, i, j):
    ref = np.load(REFERENCE)
    meta = json.loads(str(ref[f"{energy_key}__meta"]))
    rec = strip_point(meta["settings"]["energy"], i, j, meta["grid"], meta["settings"])
    assert rec["n_intervals"] == ref[f"{energy_key}__n_intervals"][i, j]
    assert rec["outer_edge"] == pytest.approx(ref[f"{energy_key}__outer_edge"][i, j], rel=B_OUT_TOLERANCE)
