"""(phi1, dphi) re-indexing of the stored maps (analysis/delta_phi.py) is exact and loses nothing."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from delta_phi import TWO_PI, delta_grid, sample_angles, wrap   # noqa: E402

REF = ROOT / "tests" / "reference"


def stored_maps():
    torus = np.load(REF / "map_torus32.npz")
    yield "torus32", "torus", torus["phi"], torus["phi"], torus["outer_edge"]
    zoom = np.load(REF / "zoom_maps.npz")
    for key in ("strip_anti", "strip_mirror", "patch_tt", "patch_min"):
        line = json.loads(str(zoom[f"{key}__meta"]))["grid"]["line"]
        yield key, line, zoom[f"{key}__along"], zoom[f"{key}__across"], zoom[f"{key}__outer_edge"]


@pytest.mark.parametrize("name,kind,along,across,values", list(stored_maps()), ids=lambda x: x if isinstance(x, str) else "")
def test_delta_grid_is_an_exact_reindexing(name, kind, along, across, values):
    p1, p2 = sample_angles(kind, along, across)
    grid, x, y, step = delta_grid(values, p1, p2)
    np.testing.assert_array_equal(x, p1[:, 0])
    assert np.isfinite(grid).sum() == np.isfinite(values).sum()          # nothing dropped or duplicated
    rows = np.round((wrap(p2 - p1) - y[0]) / step).astype(int) % int(round(TWO_PI / step))
    np.testing.assert_array_equal(np.take_along_axis(grid, rows, 1), values)
    assert np.max(np.abs(wrap(y[rows] - (p2 - p1), -np.pi))) < 1e-9       # each sample sits at its own dphi (mod 2 pi)


def test_torus_roll():
    torus = np.load(REF / "map_torus32.npz")
    v, phi = torus["outer_edge"], torus["phi"]
    grid, _, _, _ = delta_grid(v, *sample_angles("torus", phi, phi))
    n = len(phi)
    for i in range(n):
        np.testing.assert_array_equal(grid[i], np.roll(v[i], -i))       # dphi = (j - i) mod n


def test_incommensurate_grid_is_refused():
    phi1 = np.zeros((2, 5)) + np.array([[0.0], [0.3]])                   # dphi steps of 0.3 do not divide 2 pi
    phi2 = phi1 + 0.3 * np.arange(5)
    with pytest.raises(ValueError):
        delta_grid(np.ones((2, 5)), phi1, phi2)
