"""Stage 8a — zoomed strip grids across the antiparallel and mirror lines.

Checks
------
* strip coordinates: d = 0 on the antiparallel strip gives m2 = -m1 exactly; d = 0 on the
  mirror strip gives m2 = m1 reflected across the y axis; the axes have the requested steps
  (pi/64 along, pi/512 across) and contain the torus-map angles.
* consistency with the stored 32 x 32 production map: strip points that coincide with torus
  grid points reproduce its interval counts, outer edges and capture measures (rel 2e-6),
  including an escape-window point in the trough below the mirror line.
* bookkeeping: load_map returns strip arrays with coordinates; expected_points covers the strip;
  resuming with a different grid is refused; `sweep.py --strip ... --limit 0` records the grid.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from physics.edges import expected_points, in_plane_moment, load_map, strip_angles, strip_axes, strip_point

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "map_torus32.npz"
ANTI = {"kind": "strip", "line": "antiparallel", "along_pi": [0.0, 1.0, 64], "across_pi": [-0.078125, 0.078125, 81]}
MIRROR = {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, 64], "across_pi": [-0.1875, 0.09375, 145]}


def test_strip_lines_are_the_named_families():
    for s in np.linspace(0, 2 * np.pi, 7):
        p1, p2 = strip_angles("antiparallel", s, 0.0)
        np.testing.assert_allclose(in_plane_moment(p2), -in_plane_moment(p1), atol=1e-15)
        p1, p2 = strip_angles("mirror", s, 0.0)
        m1, m2 = in_plane_moment(p1), in_plane_moment(p2)
        np.testing.assert_allclose(m2, [-m1[0], m1[1], 0.0], atol=1e-15)


@pytest.mark.parametrize("grid", [ANTI, MIRROR])
def test_strip_axes_steps_and_torus_coincidence(grid):
    along, across = strip_axes(grid)
    assert len(along) == 64 and np.allclose(np.diff(along), np.pi / 64)
    assert np.allclose(np.diff(across), np.pi / 512)
    for target in (0.0, np.pi / 16, -np.pi / 16):              # torus offsets lie on the strip
        assert np.min(np.abs(across - target)) < 1e-12
    assert np.allclose(along[::4], np.pi / 16 * np.arange(16))   # every 4th along point is on the torus


@pytest.mark.parametrize("grid,i,j,torus_ij", [
    (ANTI, 16, 40, (4, 20)),          # s = pi/4, d = 0            -> (4, 20): on the antiparallel line
    (ANTI, 16, 72, (4, 21)),          # d = +pi/16                  -> one torus cell off the line
    (ANTI, 16, 8, (4, 19)),           # d = -pi/16
    (MIRROR, 16, 96, (4, 12)),        # s = pi/4, d = 0 on the mirror line
    (MIRROR, 16, 64, (4, 11)),        # d = -pi/16: the trough, an escape-window point
])
def test_strip_points_reproduce_torus_map(grid, i, j, torus_ij):
    ref = np.load(REFERENCE)
    settings = {**json.loads(str(ref["settings"])), "method": "numba"}
    along, across = strip_axes(grid)
    p1, p2 = strip_angles(grid["line"], along[i], across[j])
    ti, tj = torus_ij
    assert np.isclose(np.mod(p1, 2 * np.pi), ref["phi"][ti]) and np.isclose(np.mod(p2, 2 * np.pi), ref["phi"][tj])
    rec = strip_point(settings["energy"], i, j, grid, settings)
    assert rec["n_intervals"] == ref["n_intervals"][ti, tj]
    assert rec["outer_edge"] == pytest.approx(ref["outer_edge"][ti, tj], rel=2e-6)
    assert rec["capture_measure"] == pytest.approx(ref["capture_measure"][ti, tj], rel=2e-6)


def test_load_map_and_expected_points_for_strips(tmp_path):
    grid = {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, 4], "across_pi": [-0.1, 0.1, 3]}
    meta = {"grid": grid, "n_phi": 32, "fundamental": False, "settings": {}, "git_hash": "x", "git_dirty": False}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    with open(tmp_path / "points.jsonl", "w") as f:
        for i in range(4):
            for j in range(3):
                f.write(json.dumps({"i": i, "j": j, "outer_edge": 5.0 + i + 0.1 * j, "capture_measure": 4.0,
                                    "n_intervals": 1, "n_timeouts": 0}) + "\n")
    data = load_map(tmp_path)
    assert data["outer_edge"].shape == (4, 3)
    assert data["outer_edge"][2, 1] == 5.0 + 2 + 0.1
    np.testing.assert_allclose(data["across"], np.pi * np.array([-0.1, 0.0, 0.1]))
    np.testing.assert_allclose(data["phi2"], np.pi - data["phi1"] + data["across"][None, :])
    assert expected_points(meta) == {(i, j) for i in range(4) for j in range(3)}


def test_resume_refuses_a_different_grid(tmp_path):
    import sweep
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"settings": {"energy": 0.1}, "n_phi": 32, "fundamental": False, "grid": ANTI}))
    sweep.check_resume(meta, {"energy": 0.1}, 32, False, ANTI)
    with pytest.raises(SystemExit):
        sweep.check_resume(meta, {"energy": 0.1}, 32, False, MIRROR)
    with pytest.raises(SystemExit):
        sweep.check_resume(meta, {"energy": 0.1}, 32, False, None)      # torus vs strip


def test_sweep_strip_records_grid():
    name = "pytest_strip_limit0"
    try:
        out = subprocess.run([sys.executable, "sweep.py", "--name", name, "--strip", "mirror",
                              "--along-pi", "0", "1", "8", "--across-pi", "-0.1", "0.1", "5",
                              "--method", "numba", "--limit", "0", "--workers", "1"],
                             cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert "40 points" in out.stdout
        meta = json.loads((ROOT / "results" / name / "meta.json").read_text())
        assert meta["grid"] == {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, 8],
                                "across_pi": [-0.1, 0.1, 5]}
    finally:
        shutil.rmtree(ROOT / "results" / name, ignore_errors=True)


@pytest.mark.parametrize("key,i,j", [("strip_anti", 16, 40), ("strip_mirror", 48, 130), ("patch_tt", 48, 48)])
def test_regression_against_stored_zoom_maps(key, i, j):
    """Recompute stored stage-8 points (tests/reference/zoom_maps.npz) at their recorded settings."""
    zoom = np.load(ROOT / "tests" / "reference" / "zoom_maps.npz")
    meta = json.loads(str(zoom[f"{key}__meta"]))
    rec = strip_point(meta["settings"]["energy"], i, j, meta["grid"], meta["settings"])
    assert rec["n_intervals"] == zoom[f"{key}__n_intervals"][i, j]
    assert rec["outer_edge"] == pytest.approx(zoom[f"{key}__outer_edge"][i, j], rel=1e-9)


def test_compare_maps_rejects_identically_incomplete_runs(tmp_path):
    """Two runs truncated the same way must not PASS a directory-to-directory comparison."""
    grid = {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, 2], "across_pi": [-0.1, 0.1, 2]}
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"grid": grid, "n_phi": 32, "fundamental": False,
                                                 "settings": {"energy": 0.1}, "git_hash": "x", "git_dirty": False}))
        rec = {"i": 0, "j": 0, "outer_edge": 5.0, "capture_measure": 4.0, "n_intervals": 1, "n_timeouts": 0,
               "edges": [5.0], "edge_kinds": ["ce"], "seconds": 0.0}
        (d / "points.jsonl").write_text(json.dumps(rec) + "\n")         # 1 of 4 points
    out = subprocess.run([sys.executable, str(ROOT / "analysis" / "compare_maps.py"), str(tmp_path / "a"),
                          str(tmp_path / "b")], capture_output=True, text=True)
    assert out.returncode == 1 and "INCOMPLETE" in out.stdout


@pytest.mark.parametrize("along", [["0", "1", "2.5"], ["0", "1", "0"], ["1", "0", "8"]])
def test_sweep_rejects_bad_strip_counts(along):
    out = subprocess.run([sys.executable, "sweep.py", "--name", "pytest_bad_counts", "--strip", "mirror",
                          "--along-pi", *along, "--across-pi", "-0.1", "0.1", "5", "--limit", "0"],
                         cwd=ROOT, capture_output=True, text=True, timeout=60)
    shutil.rmtree(ROOT / "results" / "pytest_bad_counts", ignore_errors=True)
    assert out.returncode != 0 and "--along-pi" in out.stderr + out.stdout


def write_strip_run(directory, along_n, across, records_fn):
    grid = {"kind": "strip", "line": "mirror", "along_pi": [0.0, 1.0, along_n], "across_pi": across}
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps({"grid": grid, "n_phi": 32, "fundamental": False,
                                                     "settings": {"energy": 0.1}, "git_hash": "x", "git_dirty": False}))
    with open(directory / "points.jsonl", "w") as f:
        for i in range(along_n):
            for j in range(int(across[2])):
                f.write(json.dumps(records_fn(i, j)) + "\n")


def good_record(i, j):
    return {"i": i, "j": j, "outer_edge": 5.0 + 0.1 * j, "capture_measure": 4.9, "n_intervals": 1, "n_timeouts": 0,
            "edges": [5.0 + 0.1 * j], "edge_kinds": ["ce"], "contact_at_scan_top": False, "unresolved": [],
            "seconds": 0.0}


def test_unresolved_profiles_fail_completeness(tmp_path):
    """A timeout inside a profile gives a *successful* record with unresolved ranges: not complete."""
    def records(i, j):
        rec = good_record(i, j)
        if (i, j) == (1, 1):
            rec.update(unresolved=[[2.9, 3.0]], n_intervals=-1, capture_measure=None, outer_edge=None, n_timeouts=1)
        return rec
    for name in ("a", "b"):
        write_strip_run(tmp_path / name, 2, [-0.1, 0.1, 3], records)
    from physics.edges import completeness
    assert completeness(load_map(tmp_path / "a"))["unresolved"] == [(1, 1)]
    compare = subprocess.run([sys.executable, str(ROOT / "analysis" / "compare_maps.py"), str(tmp_path / "a"),
                              str(tmp_path / "b")], capture_output=True, text=True)
    assert compare.returncode == 1 and "unresolved" in compare.stdout
    checks = subprocess.run([sys.executable, str(ROOT / "analysis" / "map_checks.py"), str(tmp_path / "a")],
                            capture_output=True, text=True)
    assert checks.returncode == 1 and "INCOMPLETE" in checks.stdout


def test_strip_maps_handles_a_single_row(tmp_path):
    sys.path.insert(0, str(ROOT / "analysis"))
    import strip_maps
    write_strip_run(tmp_path / "one_row", 1, [-0.1, 0.1, 3], good_record)
    written = strip_maps.figures(tmp_path / "one_row", out_dir=tmp_path / "figures")
    assert written and all(p.exists() and p.parent == tmp_path / "figures" for p in written)


def test_resume_records_provenance(tmp_path):
    import sweep
    meta = tmp_path / "meta.json"
    old = {"git_hash": "aaa", "git_dirty": False}
    meta.write_text(json.dumps(old))
    assert sweep.record_resume(meta, dict(old), "aaa", False, 0) == old          # nothing computed: unchanged
    new = sweep.record_resume(meta, dict(old), "bbb", True, 5)
    stored = json.loads(meta.read_text())
    assert stored["git_mixed"] and stored["git_dirty"] and stored["resumes"][0]["points"] == 5 and new == stored
