"""Compare an orientation map against a stored reference map (e.g. numba vs scipy).

Usage
-----
    python analysis/compare_maps.py results/torus32_numba tests/reference/map_torus32.npz

Pass criteria (stage 7): identical contact-interval counts at every grid point, identical nan
patterns (a value missing in only one map fails), and outer edge and capture measure equal to
relative 2e-6 (twice the bisection rel_tol; the two integrators
take the same steps, so differences only arise where a bisection midpoint falls within
roundoff-amplified distance of an edge). Exits with status 1 on failure.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.edges import load_map   # noqa: E402

RTOL = 2e-6


def max_rel_diff(a, b):
    """Max relative difference; a cell that is nan in only one map counts as infinite."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    both = ~np.isnan(a)
    return float(np.max(np.abs(a[both] - b[both]) / np.abs(b[both]))) if both.any() else 0.0


def main():
    new = load_map(sys.argv[1])
    ref = np.load(sys.argv[2])
    ref_settings = json.loads(str(ref["settings"]))
    new_settings = {k: v for k, v in new["meta"]["settings"].items() if k != "method"}
    assert new_settings == {k: v for k, v in ref_settings.items() if k != "method"}, "different map settings"
    same_intervals = np.array_equal(new["n_intervals"], ref["n_intervals"])
    rel_edge = max_rel_diff(new["outer_edge"], ref["outer_edge"])
    rel_measure = max_rel_diff(new["capture_measure"], ref["capture_measure"])
    ok = same_intervals and rel_edge < RTOL and rel_measure < RTOL
    print(f"interval counts identical: {same_intervals}; max rel diff outer edge {rel_edge:.1e}, "
          f"capture measure {rel_measure:.1e} (tol {RTOL:.0e})  -> {'PASS' if ok else 'FAIL'}")
    seconds = [r["seconds"] for r in new["records"].values()]
    print(f"mean {np.mean(seconds):.2f} s per map point ({new['meta']['settings'].get('method')})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
