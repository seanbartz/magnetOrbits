"""Orientation map of capture profiles: many (phi1, phi2) points in parallel.

Each point is ONE call to physics.edges.sweep_point, which calls capture_profile, which calls
solve.run_trajectory -- the same code path as run_single.py. Any map point can be re-inspected
with, e.g.,
    python run_single.py --energy 0.033501 --b 5.75 --m1 cos(phi1) sin(phi1) 0 --m2 ...
or its whole b-profile with physics.edges.in_plane_profile(E, phi1, phi2).

Usage
-----
    python sweep.py --name torus32 --n-phi 32                       # full torus, 32 x 32
    python sweep.py --name fund64 --n-phi 64 --fundamental          # symmetry-reduced domain
    python sweep.py --name torus32 --n-phi 32 --workers 18          # resume: skips done points
    python sweep.py --name anti_strip --strip antiparallel --along-pi 0 1 64 \
                    --across-pi -0.078125 0.078125 81 --method numba   # zoomed strip (stage 8)

Strips (physics/edges.py: strip_angles): s along the line, d across it, in units of pi.
    antiparallel: phi1 = s, phi2 = s + pi + d;   mirror: phi1 = s, phi2 = pi - s + d.

Output (results/<name>/, not tracked by git):
    meta.json       parameters, solver settings, git hash + dirty flag, start time, host
    points.jsonl    one JSON object per finished point (appended as points finish, so a
                    crash loses nothing and a rerun skips completed points)
Collect into arrays with physics.edges.load_map(results/<name>).

Parallelization: independent points over ProcessPoolExecutor, BLAS pinned to 1 thread per
worker (set before numpy is imported), 18 workers by default on the 20-core Mac Studio.
Cost: ~28 s per point at n_scan = 200 with 18 concurrent workers (measured), so 1024 points
take ~27-30 min on the Mac Studio.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from physics.edges import STRIP_LINES, fundamental_domain, strip_point, sweep_point
from tests.published.bartz_shaw_2022 import ENERGY_TILDE as PAPER_ENERGY


# Everything a sweep imports at run time: physics/, this script, and tests/published (the
# default energy). A change in any of them marks the run dirty.
RUNTIME_PATHS = ("physics", "sweep.py", "tests/published")


def git_state():
    root = Path(__file__).parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", *RUNTIME_PATHS], cwd=root,
                                capture_output=True, text=True).stdout)
    return head, dirty


def check_resume(meta_file, settings, n_phi, fundamental, grid=None):
    """Refuse to resume a run whose settings, grid or run mode (full/fundamental) differ."""
    old = json.loads(meta_file.read_text())
    grid = grid or {"kind": "torus"}
    if (old["settings"] != settings or old["n_phi"] != n_phi or old["fundamental"] != fundamental
            or old.get("grid", {"kind": "torus"}) != grid):
        raise SystemExit(f"{meta_file} has different settings or run mode; use a new --name")
    return old


def completed_points(points_file):
    """(i, j) of points already computed successfully; failed points are retried on resume."""
    done = set()
    if points_file.exists():
        for line in points_file.read_text().splitlines():
            rec = json.loads(line)
            if "error" not in rec:
                done.add((rec["i"], rec["j"]))
    return done


def parse_args():
    p = argparse.ArgumentParser(description="Orientation map of capture profiles")
    p.add_argument("--name", required=True, help="results/<name>/")
    p.add_argument("--energy", type=float, default=PAPER_ENERGY, help="E_tilde (default: paper, 0.033501)")
    p.add_argument("--n-phi", type=int, default=32, help="grid points per angle on [0, 2 pi)")
    p.add_argument("--fundamental", action="store_true", help="only the symmetry-reduced domain")
    p.add_argument("--strip", choices=STRIP_LINES, help="zoomed strip across this line instead of the torus")
    p.add_argument("--along-pi", type=float, nargs=3, metavar=("LO", "HI", "N"),
                   help="strip: along-line coordinate s in [LO, HI) * pi, N points")
    p.add_argument("--across-pi", type=float, nargs=3, metavar=("LO", "HI", "M"),
                   help="strip: across-line offset d in [LO, HI] * pi, M points (ends included)")
    p.add_argument("--b-min", type=float, default=0.05)
    p.add_argument("--b-max", type=float, default=10.0)
    p.add_argument("--n-scan", type=int, default=200)
    p.add_argument("--rel-tol", type=float, default=1e-6)
    # r0: with the first-order in-state correction the edges converge as r0^-2. At E = 0.0335
    # r0 = 1000 leaves up to 5e-4 relative error near the low-edge "diamond"; r0 = 64000 leaves
    # <= 3e-7 (Richardson, measured), below rel_tol, for ~15% more cost.
    p.add_argument("--start-radius", type=float, default=64000.0)
    p.add_argument("--method", choices=("reference", "numba"), default="reference",
                   help="numba: compiled DOP853 (physics/fast.py), ~300x faster per trajectory")
    p.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 1))
    p.add_argument("--limit", type=int, default=None, help="only the first N pending points (smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(__file__).parent / "results" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    points_file = out_dir / "points.jsonl"

    # 1. The grid. Torus: index pairs (i, j) -> phi = 2 pi i / n; odd n breaks the symmetry
    #    group (phi + pi must be a grid point), which map_checks and load_map rely on.
    #    Strip: (i along, j across) -> strip_angles(line, s_i, d_j).
    n = args.n_phi
    if args.strip:
        if args.along_pi is None or args.across_pi is None or args.fundamental:
            raise SystemExit("--strip needs --along-pi and --across-pi (and no --fundamental)")
        for label, spec in (("--along-pi", args.along_pi), ("--across-pi", args.across_pi)):
            if spec[2] != int(spec[2]) or spec[2] < (1 if label == "--along-pi" else 2):
                raise SystemExit(f"{label}: the point count must be a positive integer (>= 2 across), got {spec[2]}")
            if not spec[1] > spec[0]:
                raise SystemExit(f"{label}: need HI > LO, got {spec[0]} {spec[1]}")
        grid = {"kind": "strip", "line": args.strip,
                "along_pi": [args.along_pi[0], args.along_pi[1], int(args.along_pi[2])],
                "across_pi": [args.across_pi[0], args.across_pi[1], int(args.across_pi[2])]}
        pairs = [(i, j) for i in range(grid["along_pi"][2]) for j in range(grid["across_pi"][2])]
    else:
        if n % 2:
            raise SystemExit("--n-phi must be even (phi + pi must lie on the grid)")
        grid = {"kind": "torus"}
        pairs = fundamental_domain(n) if args.fundamental else [(i, j) for i in range(n) for j in range(n)]
    settings = dict(energy=args.energy, b_min=args.b_min, b_max=args.b_max, n_scan=args.n_scan,
                    rel_tol=args.rel_tol, start_radius=args.start_radius, method=args.method)

    # 2. Metadata (refuse to mix settings with an existing run).
    head, dirty = git_state()
    meta = dict(settings=settings, n_phi=n, fundamental=args.fundamental, grid=grid, git_hash=head,
                git_dirty=dirty, solver="run_trajectory: r0=settings.start_radius, rtol=1e-10, atol=1e-12, "
                "beta=0.4, asymptotic_correction=True, stop='escape'",
                started=time.strftime("%Y-%m-%d %H:%M:%S"), host=platform.node())
    meta_file = out_dir / "meta.json"
    if meta_file.exists():
        old = check_resume(meta_file, settings, n, args.fundamental, grid)
        if old["git_hash"] != head:
            print(f"warning: resuming a run started at {old['git_hash'][:8]} with code at {head[:8]}")
    else:
        meta_file.write_text(json.dumps(meta, indent=2))

    # 3. Resume: skip points already computed successfully (failed points are retried).
    done = completed_points(points_file)
    pending = [(i, j) for i, j in pairs if (i, j) not in done]
    if args.limit is not None:           # --limit 0 means "run nothing" (e.g. just write meta.json)
        pending = pending[:args.limit]
    print(f"{len(pairs)} points, {len(done)} done, running {len(pending)} on {args.workers} workers "
          f"(git {head[:8]}{', DIRTY' if dirty else ''})")

    # 4. Parallel over independent points; append each result as it finishes.
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(points_file, "a") as sink:
        if grid["kind"] == "strip":
            futures = {pool.submit(strip_point, args.energy, i, j, grid, settings): (i, j) for i, j in pending}
        else:
            futures = {pool.submit(sweep_point, args.energy, i, j, n, settings): (i, j) for i, j in pending}
        for count, fut in enumerate(as_completed(futures), 1):
            i, j = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:             # one bad point must not kill the sweep
                rec = dict(i=i, j=j, n_phi=n, error=repr(exc))
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
            if count % 25 == 0 or count == len(pending):
                elapsed = time.time() - t0
                print(f"  {count}/{len(pending)}  {elapsed/60:.1f} min elapsed, "
                      f"~{elapsed/count*(len(pending)-count)/60:.1f} min left", flush=True)


if __name__ == "__main__":      # required on macOS (spawn start method)
    main()
