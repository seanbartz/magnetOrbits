"""Convergence check of an orientation map: recompute sample points at tighter settings.

Usage
-----
    python analysis/map_convergence.py results/torus32 [--n-random 10] [--workers 18]

For each sampled point, the capture profile is recomputed with every numerical knob tightened
at once -- rtol 1e-10 -> 1e-12 (atol 1e-12 -> 1e-14), r0 doubled, scan 200 -> 400 points --
and compared with the stored map entry:
  * the interval structure (edge count and kinds) must be identical: the scan resolution was
    sufficient and no window was missed or invented;
  * edges must agree to about the bisection tolerance (rel 1e-6 -> threshold 3e-6), which is
    above the integration error (~1e-10) and, at the production r0 = 64000, the r0 error
    (<= 3e-7; it falls as r0^-2 and was 5e-4 at r0 = 1000 near the diamond, E = 0.0335).
The sample is every multi-interval point in the fundamental domain plus random others
(seed recorded). Unlike the symmetry check, this tests numerical accuracy.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.edges import fundamental_domain, in_plane_profile, load_map   # noqa: E402

TIGHT_TOLERANCES = dict(rtol=1e-12, atol=1e-14)
EDGE_THRESHOLD = 3e-6


def recompute(args):
    energy, phi1, phi2, settings = args
    return in_plane_profile(energy, phi1, phi2, b_min=settings["b_min"], b_max=settings["b_max"],
                            n_scan=2 * settings["n_scan"], rel_tol=settings["rel_tol"],
                            start_radius=2 * settings.get("start_radius", 1000.0), **TIGHT_TOLERANCES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_dir")
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 1))
    args = p.parse_args()

    data = load_map(args.result_dir, expand_symmetry=False)
    n, settings = data["meta"]["n_phi"], data["meta"]["settings"]
    reps = [r for r in fundamental_domain(n) if r in data["records"]]
    multi = [r for r in reps if data["records"][r]["n_intervals"] != 1]
    single = [r for r in reps if r not in multi]
    rng = np.random.default_rng(args.seed)
    sample = multi + [single[k] for k in rng.choice(len(single), size=min(args.n_random, len(single)), replace=False)]
    print(f"recomputing {len(sample)} points ({len(multi)} multi-interval) with {TIGHT_TOLERANCES}, "
          f"r0 {2 * settings.get('start_radius', 1000.0):g}, n_scan {2 * settings['n_scan']} (seed {args.seed})")

    jobs = [(settings["energy"], data["records"][r]["phi1"], data["records"][r]["phi2"], settings) for r in sample]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tight = list(pool.map(recompute, jobs))

    worst, failures = 0.0, []
    for (i, j), prof in zip(sample, tight):
        rec = data["records"][(i, j)]
        # Whole profile structure: edge kinds, interval count, scan-top contact, undecided parts.
        same = (prof["edge_kinds"] == rec["edge_kinds"] and prof["n_intervals"] == rec["n_intervals"]
                and prof["contact_at_scan_top"] == rec["contact_at_scan_top"]
                and not prof["unresolved"] and not rec.get("unresolved"))
        diff = np.max(np.abs(np.array(prof["edges"]) - rec["edges"]) / np.array(rec["edges"])) if same and rec["edges"] else np.nan
        if not same or diff > EDGE_THRESHOLD:
            failures.append((i, j))
        if same and rec["edges"]:
            worst = max(worst, diff)
        print(f"  ({i:2d},{j:2d}) intervals {rec['n_intervals']} -> {prof['n_intervals']}  "
              f"max rel edge diff {diff:.1e}")
    status = "PASS" if not failures else f"FAIL at {failures}"
    print(f"structure identical at {len(sample) - len(failures)}/{len(sample)}; max rel edge diff {worst:.1e} "
          f"(threshold {EDGE_THRESHOLD:.0e})  -> {status}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
