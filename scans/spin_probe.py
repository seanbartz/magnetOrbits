"""How much does a non-zero initial spin matter?

Every scan in `notes/findings.md` uses a non-rotating in-state (s1 = s2 = 0).  This
probes the size of the effect of relaxing that, holding the *translational* energy fixed
so the locked-dipole baseline btil_max is unchanged and any movement of the capture edge
is attributable to spin alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from magdip import locked, scatter  # noqa: E402
from survey import capture_intervals  # noqa: E402


def _job(args):
    E, p1, p2, s1, s2, n, rho0 = args
    bm = locked.b_max(E)
    b = np.linspace(0.05 * bm, 2.6 * bm, n)
    r = scatter.run_batch(E, b, p1, p2, rho0=rho0, s1=s1, s2=s2,
                          regularized=True, rtol=1e-11)
    iv = capture_intervals(b, r["collapsed"])
    edge = max((h for _, h in iv), default=np.nan)
    return (s1, s2, len(iv), sum(h - l for l, h in iv), edge / bm,
            float(np.nanmax(r["delay"])) if r["escaped"].any() else np.nan)


def main(E=0.3, orient=(1.9, 1.3), n=8000, rho0=200.0, workers=20):
    spins = [(0.0, 0.0), (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1),
             (0.3, 0.3), (-0.3, -0.3), (0.3, -0.3), (0.5, 0.2)]
    jobs = [(E, orient[0], orient[1], s1, s2, n, rho0) for s1, s2 in spins]
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        rows = pool.map(_job, jobs)
    print("Etil_trans=%g  phi0=(%.2f,%.2f)  N=%d  rho0=%g  locked btil_max=%.4f"
          % (E, orient[0], orient[1], n, rho0, locked.b_max(E)))
    print("%12s %8s %10s %11s %10s %10s"
          % ("(s1,s2)", "nIval", "measure", "edge/bmax", "maxDelay", "spinKE/E"))
    for (s1, s2, ni, meas, edge, md) in rows:
        print("%12s %8d %10.4f %11.4f %10.1f %10.3f"
              % ("(%.1f,%.1f)" % (s1, s2), ni, meas, edge, md,
                 0.5 * (s1 ** 2 + s2 ** 2) / E))
    return rows


if __name__ == "__main__":
    main()
