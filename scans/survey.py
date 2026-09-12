"""Survey the deflection function over Etil and initial orientation.

Counts the number of disjoint capture intervals in a chi(btil) line scan -- the
cheapest signature of a non-smooth (potentially fractal) capture boundary.  A regular
scattering regime gives exactly one interval; a fractal one gives a count that keeps
growing as the resolution increases.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from magdip import locked, scatter  # noqa: E402


def capture_intervals(b, collapsed):
    """Contiguous runs of capture, as ``(b_lo, b_hi)`` pairs."""
    c = np.asarray(collapsed, dtype=int)
    if c.size == 0:
        return []
    out = []
    start = 0
    for k in range(1, c.size):
        if c[k] != c[k - 1]:
            if c[start]:
                out.append((b[start], b[k - 1]))
            start = k
    if c[start]:
        out.append((b[start], b[-1]))
    return out


def line_scan(E, phi1, phi2, n=20_000, span=(0.05, 2.6), **kw):
    bm = locked.b_max(E)
    b = np.linspace(span[0] * bm, span[1] * bm, n)
    kw.setdefault("regularized", True)
    kw.setdefault("rtol", 1e-11)
    r = scatter.run_batch(E, b, phi1, phi2, **kw)
    r["b"] = b
    r["b_max_locked"] = bm
    r["intervals"] = capture_intervals(b, r["collapsed"])
    return r


def summarise(r):
    iv = r["intervals"]
    bm = r["b_max_locked"]
    measure = sum(hi - lo for lo, hi in iv)
    edge = max((hi for _, hi in iv), default=np.nan)
    return {"n_intervals": len(iv), "measure": measure, "edge": edge,
            "edge_over_bmax": edge / bm, "measure_over_2bmax": measure / (2 * bm),
            "n_timeout": int((r["outcome"] == scatter.TIMEOUT).sum()),
            "max_delay": float(np.nanmax(r["delay"])) if r["escaped"].any() else np.nan}


def _job(args):
    E, p1, p2, n = args
    return (E, p1, p2) + tuple(summarise(line_scan(E, p1, p2, n=n))[k] for k in
                               ("n_intervals", "measure", "edge_over_bmax",
                                "n_timeout", "max_delay"))


def main(energies=None, orientations=None, n=20_000, workers=20):
    import itertools
    import multiprocessing as mp
    energies = energies or [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    orientations = orientations or [(0.0, 0.0), (0.7, -2.1), (1.9, 1.3),
                                    (np.pi / 2, -np.pi / 2)]
    jobs = [(E, p1, p2, n) for E, (p1, p2) in
            itertools.product(energies, orientations)]
    with mp.Pool(workers) as pool:
        rows = pool.map(_job, jobs)
    print("%6s %16s %6s %10s %10s %8s %10s"
          % ("Etil", "(phi1,phi2)", "nIval", "measure", "edge/bmax", "nTO", "maxDelay"))
    for E, p1, p2, ni, meas, edge, nto, md in rows:
        print("%6.2f %16s %6d %10.4f %10.4f %8d %10.1f"
              % (E, "(%.2f,%.2f)" % (p1, p2), ni, meas, edge, nto, md))
    return rows


if __name__ == "__main__":
    main()
