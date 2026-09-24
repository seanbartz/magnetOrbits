"""Capture profile in impact parameter for one orientation: all contact/escape edges.

Place in the program
--------------------
    sweep.py (orientation maps) / run_profile.py / tests
        -> edges.capture_profile(E, m1, m2, ...)
            -> capture.outcome_at -> solve.run_trajectory(stop="escape")   (one code path)

What it computes
----------------
For fixed E_tilde and asymptotic moment directions m1, m2, the outcome is a function of b
alone: "contact" (reaches r = 2) or "escape" (rigorous criterion). capture.find_capture_edge
assumes a single contact interval; here the number of intervals is not assumed, but the
resolution is (see "one transition per scan cell" below):

  1. scan b on a uniform grid over [b_min, b_max] (n_scan points);
  2. bisect every adjacent pair of scan points whose outcomes differ, to relative width
     rel_tol -- one edge per such pair;
  3. report the sorted edges, the contact intervals they bound, their total length
     ("capture measure" over [b_min, b_max]), the outer edge (largest contact->escape
     edge) and the interval count.

Limits of the method (report, don't hide)
-----------------------------------------
* ONE TRANSITION PER SCAN CELL is assumed. A cell whose end outcomes differ may contain any
  odd number of transitions and yields one edge; a cell whose ends agree may hide any even
  number. No finite sampling can exclude this, so edges and interval counts are those
  resolved at the scan spacing (b_max - b_min)/(n_scan - 1), and interval counts are lower
  bounds. The assumption is tested, not trusted: a map is accepted only if refining the scan
  (n_scan x2 on sampled points in analysis/map_convergence.py; x4 on every point of the
  production map, PLAN stage 7) leaves every profile's structure unchanged.
* The measure is over [b_min, b_max], not [0, inf): b < b_min is not sampled (b = 0 need
  not collide: side-by-side parallel moments repel), and contact at b_max is flagged via
  `contact_at_scan_top` rather than silently truncated.
* A timeout (long-lived complex) is never turned into an edge. At a scan point it is recorded
  as "t"; inside a bisection bracket the bracket is recorded as unresolved. If anything is
  undecided, the interval list, interval count and capture measure are reported as undecided
  (None / -1 / nan), and the outer edge only if it lies above every undecided location.

Also defines the in-plane parameterization used by the 6b maps:
    m_i = (cos phi_i, sin phi_i, 0),  phi measured from the incoming direction +x at t -> -inf.
Symmetries (see tests/test_edges.py): global flip (phi1+pi, phi2+pi) and, combined with
particle exchange, (phi2, phi1) give the same profile.
"""

import numpy as np

from physics.solve import run_trajectory

OUTCOME_CODE = {"contact": "c", "escape": "e", "timeout": "t", "undecided": "t"}


def in_plane_moment(phi):
    """Unit moment in the scattering (xy) plane at angle phi from +x."""
    return np.array([np.cos(phi), np.sin(phi), 0.0])


def capture_profile(energy, m1, m2, b_min=0.05, b_max=10.0, n_scan=200, rel_tol=1e-6,
                    **trajectory_kwargs):
    """All contact/escape edges in b for one orientation (see module docstring).

    Returns dict with
        edges               sorted array of edge positions
        edge_kinds          "ce" (contact below, escape above) or "ec", per edge
        intervals           list of (lo, hi) contact intervals within [b_min, b_max]
                            (None if anything is undecided)
        capture_measure     total length of the contact intervals (nan if undecided)
        outer_edge          largest "ce" edge (nan if nothing collides, or if an undecided
                            location lies above it)
        n_intervals         number of contact intervals (-1 if undecided)
        unresolved          list of (lo, hi) b-ranges left undecided by timeouts
        scan_codes          string of outcome codes on the scan grid ("c", "e", "t")
        contact_at_scan_top True if b_max still collides (edge beyond the scan)
        n_timeouts          timeouts anywhere (scan grid and bisection), n_runs
    """
    n_runs = 0
    n_timeouts = 0

    def code(b):
        nonlocal n_runs, n_timeouts
        n_runs += 1
        res = run_trajectory(energy, b, m1, m2, stop="escape", **trajectory_kwargs)
        c = OUTCOME_CODE[res["outcome"]]
        n_timeouts += c == "t"
        return c

    # 1. Uniform scan.
    grid = np.linspace(b_min, b_max, n_scan)
    codes = [code(b) for b in grid]

    # 2. Bisect every c/e sign change. A timeout at a midpoint leaves the bracket
    #    unresolved (recorded, never turned into an edge).
    edges, kinds = [], []
    unresolved = [(grid[max(k - 1, 0)], grid[min(k + 1, n_scan - 1)])
                  for k, c in enumerate(codes) if c == "t"]
    for k in range(n_scan - 1):
        lo_code, hi_code = codes[k], codes[k + 1]
        if {lo_code, hi_code} != {"c", "e"}:
            continue
        lo, hi = grid[k], grid[k + 1]
        resolved = True
        while hi - lo > rel_tol * 0.5 * (lo + hi):
            mid = 0.5 * (lo + hi)
            mid_code = code(mid)
            if mid_code == lo_code:
                lo = mid
            elif mid_code == hi_code:
                hi = mid
            else:
                resolved = False
                break
        if resolved:
            edges.append(0.5 * (lo + hi))
            kinds.append(lo_code + hi_code)
        else:
            unresolved.append((lo, hi))

    # 3. Contact intervals bounded by the edges (and by the scan ends), only if nothing is
    #    undecided; otherwise the interval structure is unknown and reported as such.
    ce_edges = [e for e, k in zip(edges, kinds) if k == "ce"]
    outer_edge = max(ce_edges) if ce_edges else np.nan
    if unresolved:
        intervals, capture_measure, n_intervals = None, np.nan, -1
        if ce_edges and outer_edge < max(hi for _, hi in unresolved):
            outer_edge = np.nan
    else:
        intervals = []
        start = b_min if codes[0] == "c" else None
        for edge, kind in zip(edges, kinds):
            if kind == "ec":
                start = edge
            elif start is not None:
                intervals.append((start, edge))
                start = None
        if start is not None:
            intervals.append((start, b_max))
        capture_measure = float(sum(hi - lo for lo, hi in intervals))
        n_intervals = len(intervals)

    return {
        "edges": np.array(edges), "edge_kinds": kinds, "intervals": intervals,
        "capture_measure": capture_measure, "outer_edge": outer_edge,
        "n_intervals": n_intervals, "unresolved": unresolved, "scan_codes": "".join(codes),
        "contact_at_scan_top": codes[-1] == "c",
        "n_timeouts": n_timeouts, "n_runs": n_runs,
    }


def in_plane_profile(energy, phi1, phi2, **kwargs):
    """capture_profile for in-plane asymptotic moments at angles (phi1, phi2)."""
    return capture_profile(energy, in_plane_moment(phi1), in_plane_moment(phi2), **kwargs)


# ---------------------------------------------------------------------------------------
# Orientation-map support (sweep.py): grid indices, symmetry reduction, one point, loading
# ---------------------------------------------------------------------------------------
# Grid: phi_i = 2 pi i / n, i = 0..n-1 (n even so that phi + pi is on the grid).
# Symmetry group of order 4 acting on index pairs:
#     F: (i, j) -> (i + n/2, j + n/2)   global flip m -> -m (U is bilinear in the moments)
#     S: (i, j) -> (j, i)               particle exchange composed with F: exchanging the
#                                       spheres maps the in-state onto itself after a rotation
#                                       by pi about z, which sends phi -> phi + pi.

def symmetry_orbit(i, j, n):
    """The (up to 4) index pairs equivalent to (i, j). Requires even n (phi + pi on the grid)."""
    if n % 2:
        raise ValueError("n_phi must be even (phi + pi must lie on the grid)")
    h = n // 2
    flip = ((i + h) % n, (j + h) % n)
    return {(i, j), flip, (j, i), (flip[1], flip[0])}


def fundamental_domain(n):
    """One representative (lexicographically smallest) per symmetry orbit."""
    if n % 2:
        raise ValueError("n_phi must be even (phi + pi must lie on the grid)")
    return sorted({min(symmetry_orbit(i, j, n)) for i in range(n) for j in range(n)})


def sweep_point(energy, i, j, n_phi, settings):
    """One orientation-map point as a JSON-serializable dict (the sweep.py worker)."""
    import time
    phi1, phi2 = 2 * np.pi * i / n_phi, 2 * np.pi * j / n_phi
    t0 = time.time()
    prof = in_plane_profile(energy, phi1, phi2, b_min=settings["b_min"], b_max=settings["b_max"],
                            n_scan=settings["n_scan"], rel_tol=settings["rel_tol"],
                            start_radius=settings.get("start_radius", 1000.0))
    return {
        "i": i, "j": j, "n_phi": n_phi, "phi1": phi1, "phi2": phi2,
        "outer_edge": None if np.isnan(prof["outer_edge"]) else float(prof["outer_edge"]),
        "capture_measure": None if np.isnan(prof["capture_measure"]) else prof["capture_measure"],
        "n_intervals": prof["n_intervals"], "unresolved": [list(u) for u in prof["unresolved"]],
        "edges": [float(e) for e in prof["edges"]], "edge_kinds": prof["edge_kinds"],
        "scan_codes": prof["scan_codes"], "contact_at_scan_top": prof["contact_at_scan_top"],
        "n_timeouts": prof["n_timeouts"], "n_runs": prof["n_runs"], "seconds": time.time() - t0,
    }


def load_map(result_dir, expand_symmetry=True):
    """Read results/<name>/ into (n_phi, n_phi) arrays indexed [i, j].

    With expand_symmetry, points computed on the fundamental domain are copied to every
    member of their symmetry orbit. Missing points are nan; failed points are listed.
    """
    import json
    from pathlib import Path
    result_dir = Path(result_dir)
    meta = json.loads((result_dir / "meta.json").read_text())
    n = meta["n_phi"]
    outer = np.full((n, n), np.nan)
    measure = np.full((n, n), np.nan)
    n_intervals = np.full((n, n), -1)
    timeouts = np.zeros((n, n), dtype=int)
    computed = np.zeros((n, n), dtype=bool)
    records, errors = {}, []
    for line in (result_dir / "points.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if "error" in rec:
            errors.append(rec)
            continue
        records[(rec["i"], rec["j"])] = rec   # a retried point's success supersedes its error
    errors = [e for e in errors if (e["i"], e["j"]) not in records]
    for (i, j), rec in records.items():
        targets = symmetry_orbit(i, j, n) if expand_symmetry else {(i, j)}
        for a, b in targets:
            if (a, b) in records and (a, b) != (i, j):
                continue          # computed directly: keep its own value
            outer[a, b] = np.nan if rec["outer_edge"] is None else rec["outer_edge"]
            measure[a, b] = np.nan if rec["capture_measure"] is None else rec["capture_measure"]
            n_intervals[a, b] = rec["n_intervals"]
            timeouts[a, b] = rec["n_timeouts"]
        computed[i, j] = True
    phi = 2 * np.pi * np.arange(n) / n
    return {"meta": meta, "phi": phi, "outer_edge": outer, "capture_measure": measure,
            "n_intervals": n_intervals, "n_timeouts": timeouts, "computed": computed,
            "records": records, "errors": errors}
