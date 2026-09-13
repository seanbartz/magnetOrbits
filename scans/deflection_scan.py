"""First coarse deflection-function scan: chi(btil) at fixed Etil, phi1, phi2.

Usage::

    python scans/deflection_scan.py --E 0.1 --n 20000 --out data/chi_E0.1.npz
    python scans/deflection_scan.py --E 0.1 --bmin 3.20 --bmax 3.24 --n 20000 \
        --out data/chi_E0.1_zoom.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from magdip import locked  # noqa: E402
from runner import Scan  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--E", type=float, default=0.1)
    ap.add_argument("--phi1", type=float, default=0.7)
    ap.add_argument("--phi2", type=float, default=-2.1)
    ap.add_argument("--bmin", type=float, default=None)
    ap.add_argument("--bmax", type=float, default=None)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--rho0", type=float, default=200.0)
    ap.add_argument("--rho-capture", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-11)
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    bm = locked.b_max(a.E)
    bmin = 0.05 * bm if a.bmin is None else a.bmin
    bmax = 2.20 * bm if a.bmax is None else a.bmax
    b = np.linspace(bmin, bmax, a.n)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    scan = Scan(a.out, a.E, b, a.phi1, a.phi2, chunk=a.chunk,
                meta={"E": a.E, "phi1": a.phi1, "phi2": a.phi2,
                      "bmin": bmin, "bmax": bmax, "b_max_locked": bm,
                      "rho0": a.rho0, "rho_capture": a.rho_capture},
                rho0=a.rho0, rho_capture=a.rho_capture, rtol=a.rtol,
                regularized=True)
    print("chi(b) scan: E=%g phi=(%g,%g)  b in [%.4f, %.4f]  N=%d  "
          "(locked b_max=%.4f)" % (a.E, a.phi1, a.phi2, bmin, bmax, a.n, bm))
    res = scan.run(workers=a.workers)
    esc = res["outcome"] == 1
    print("escape %d  collapse %d  timeout %d"
          % (esc.sum(), (res["outcome"] == 2).sum(), (res["outcome"] == 3).sum()))
    print("wrote %s" % a.out)
    return res


if __name__ == "__main__":
    main()
