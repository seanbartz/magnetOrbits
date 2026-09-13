"""Resolution-refinement test for a capture boundary.

Rescanning the same window of impact parameters at successively finer resolution and
counting the disjoint capture intervals distinguishes a *regular* boundary (count
saturates) from a *singular / fractal* one (count keeps growing).
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scans"))

from magdip import scatter  # noqa: E402
from survey import capture_intervals  # noqa: E402


def _chunk(args):
    E, b, p1, p2, kw = args
    r = scatter.run_batch(E, b, p1, p2, **kw)
    return np.asarray(r["outcome"])


def scan_window(E, p1, p2, lo, hi, n, workers=20, **kw):
    """Outcome codes on a uniform grid of ``n`` impact parameters in ``[lo, hi]``."""
    kw.setdefault("regularized", True)
    kw.setdefault("rtol", 1e-11)
    b = np.linspace(lo, hi, n)
    if workers <= 1 or n < 4000:
        return b, _chunk((E, b, p1, p2, kw))
    import multiprocessing as mp
    parts = np.array_split(b, workers * 2)
    with mp.Pool(workers) as pool:
        outs = pool.map(_chunk, [(E, p, p1, p2, kw) for p in parts])
    return b, np.concatenate(outs)


def refine(E, p1, p2, lo, hi, ns=(5_000, 20_000, 80_000, 320_000), workers=20, **kw):
    """Interval count and capture measure vs resolution."""
    rows = []
    for n in ns:
        b, out = scan_window(E, p1, p2, lo, hi, n, workers=workers, **kw)
        iv = capture_intervals(b, out == scatter.COLLAPSED)
        rows.append({"n": n, "db": b[1] - b[0], "n_intervals": len(iv),
                     "measure": sum(h - l for l, h in iv),
                     "n_captured": int((out == scatter.COLLAPSED).sum()),
                     "n_timeout": int((out == scatter.TIMEOUT).sum())})
    return rows


def report(rows):
    print("%10s %10s %8s %12s %10s %8s"
          % ("N", "db", "nIval", "measure", "nCapture", "nTO"))
    for r in rows:
        print("%10d %10.2e %8d %12.6e %10d %8d"
              % (r["n"], r["db"], r["n_intervals"], r["measure"], r["n_captured"],
                 r["n_timeout"]))
    ns = np.array([r["n"] for r in rows], float)
    ni = np.array([r["n_intervals"] for r in rows], float)
    if np.all(ni > 0) and len(rows) > 2:
        p = np.polyfit(np.log(ns), np.log(ni), 1)[0]
        print("\nintervals ~ N^%.3f  (0 = regular/saturating, ~1 = grid-limited "
              "singular set)" % p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=float, required=True)
    ap.add_argument("--phi1", type=float, required=True)
    ap.add_argument("--phi2", type=float, required=True)
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[5_000, 20_000, 80_000, 320_000])
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    print("window b in [%.6f, %.6f]  E=%g  phi=(%g,%g)"
          % (a.lo, a.hi, a.E, a.phi1, a.phi2))
    report(refine(a.E, a.phi1, a.phi2, a.lo, a.hi, tuple(a.ns), workers=a.workers))
