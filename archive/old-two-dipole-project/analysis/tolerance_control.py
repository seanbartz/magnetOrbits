"""Control: is the fine structure dynamics, or integration noise?

Chaotic trajectories amplify local truncation error exponentially, so a capture
boundary resolved at grid spacings of 1e-7 is only meaningful if the outcome
classification is itself converged.  This rescans the same grid at several tolerances
(and at a different capture radius) and reports the fraction of initial conditions whose
*outcome* changes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from refine import scan_window  # noqa: E402


def control(E, p1, p2, lo, hi, n=20_000, workers=20):
    base = None
    rows = []
    settings = [
        ("rtol 1e-11, rho_c 1e-2", dict(rtol=1e-11, rho_capture=1e-2)),
        ("rtol 1e-13, rho_c 1e-2", dict(rtol=1e-13, atol=1e-15, rho_capture=1e-2)),
        ("rtol 1e-13, rho_c 1e-4", dict(rtol=1e-13, atol=1e-15, rho_capture=1e-4)),
        ("unregularized, 1e-13  ", dict(rtol=1e-13, atol=1e-15, rho_capture=1e-2,
                                        regularized=False)),
    ]
    for label, kw in settings:
        _, out = scan_window(E, p1, p2, lo, hi, n, workers=workers, **kw)
        if base is None:
            base = out
            rows.append((label, 0.0, int((out == 2).sum())))
        else:
            rows.append((label, float(np.mean(out != base)), int((out == 2).sum())))
    return rows


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=float, required=True)
    ap.add_argument("--phi1", type=float, required=True)
    ap.add_argument("--phi2", type=float, required=True)
    ap.add_argument("--lo", type=float, required=True)
    ap.add_argument("--hi", type=float, required=True)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    print("outcome-stability control, N=%d, b in [%.6f, %.6f]" % (a.n, a.lo, a.hi))
    print("%-26s %14s %10s" % ("setting", "disagreement", "nCapture"))
    for label, frac, ncap in control(a.E, a.phi1, a.phi2, a.lo, a.hi, a.n, a.workers):
        print("%-26s %13.4f%% %10d" % (label, 100 * frac, ncap))
