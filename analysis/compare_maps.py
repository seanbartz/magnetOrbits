"""Compare an orientation map against a stored reference map (e.g. numba vs scipy).

Usage
-----
    python analysis/compare_maps.py results/torus32_numba tests/reference/map_torus32.npz
    python analysis/compare_maps.py results/torus32_scan800 tests/reference/map_torus32.npz --scan-refinement
    python analysis/compare_maps.py results/strip_anti_scan800 results/strip_anti --scan-refinement

The reference may be a stored .npz map or another results directory. With two directories
the grids must match, and every point's full edge list is compared as well (same number and
kinds of edges, positions to RTOL), not only the summary arrays.

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

from physics.edges import completeness_problem, load_map   # noqa: E402

RTOL = 2e-6


def max_rel_diff(a, b):
    """Max relative difference |a - b| / |b| over cells present in both maps.

    A cell that is nan in only one map, or zero in one and nonzero in the other, counts as
    infinite; equal cells (including 0 == 0, e.g. a zero capture measure) count as zero.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    both = ~np.isnan(a)
    a, b = a[both], b[both]
    if a.size == 0:
        return 0.0
    diff = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(diff == 0.0, 0.0, diff / np.abs(b))   # b == 0 != a  ->  inf
    return float(np.max(rel))


def compare_edge_lists(new_records, ref_records):
    """(number of points whose edge kinds differ, max relative edge-position difference)."""
    kind_mismatch, worst = 0, 0.0
    for key, ref_rec in ref_records.items():
        rec = new_records[key]
        if rec["edge_kinds"] != ref_rec["edge_kinds"]:
            kind_mismatch += 1
        elif ref_rec["edges"]:
            worst = max(worst, max_rel_diff(rec["edges"], ref_rec["edges"]))
    return kind_mismatch, worst


def require_complete(data, label):
    """Exit 1 unless the run has every expected point, no failures and no unresolved profiles."""
    problem = completeness_problem(data)
    if problem:
        print(f"INCOMPLETE {label}: {problem}  -> FAIL")
        raise SystemExit(1)


def main():
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    new = load_map(paths[0])
    require_complete(new, paths[0])
    ref_is_dir = Path(paths[1]).is_dir()
    if ref_is_dir:
        ref = load_map(paths[1])
        require_complete(ref, paths[1])
        assert new["grid"] == ref["grid"], f"different grids: {new['grid']} vs {ref['grid']}"
        ref_settings = ref["meta"]["settings"]
    else:
        ref = np.load(paths[1])
        ref_settings = json.loads(str(ref["settings"]))
    # --scan-refinement: the new map may use a finer b scan (n_scan) than the reference; that is
    # the whole-map test of the one-transition-per-scan-cell assumption (physics/edges.py).
    ignore = {"method", "n_scan"} if "--scan-refinement" in sys.argv else {"method"}
    new_settings = {k: v for k, v in new["meta"]["settings"].items() if k not in ignore}
    assert new_settings == {k: v for k, v in ref_settings.items() if k not in ignore}, "different map settings"
    same_intervals = np.array_equal(new["n_intervals"], ref["n_intervals"])
    rel_edge = max_rel_diff(new["outer_edge"], ref["outer_edge"])
    rel_measure = max_rel_diff(new["capture_measure"], ref["capture_measure"])
    ok = same_intervals and rel_edge < RTOL and rel_measure < RTOL
    if ref_is_dir:
        if set(new["records"]) != set(ref["records"]):
            print("the two runs do not contain the same points  -> FAIL")
            raise SystemExit(1)
        kind_mismatch, rel_all = compare_edge_lists(new["records"], ref["records"])
        ok = ok and kind_mismatch == 0 and rel_all < RTOL
        print(f"all edges: kinds differ at {kind_mismatch} points; max rel position diff {rel_all:.1e}")
    print(f"interval counts identical: {same_intervals}; max rel diff outer edge {rel_edge:.1e}, "
          f"capture measure {rel_measure:.1e} (tol {RTOL:.0e})  -> {'PASS' if ok else 'FAIL'}")
    seconds = [r["seconds"] for r in new["records"].values()]
    print(f"mean {np.mean(seconds):.2f} s per map point ({new['meta']['settings'].get('method')})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
