"""Stage 8A: uncertainty exponent and resolution scaling at selected orientations.

Usage
-----
    python analysis/uncertainty.py [--samples 50000] [--workers 18]

Points (asymptotic in-plane angles via physics.edges.strip_angles, E = paper energy, r0 = 64000,
numba). The two controls have a single smooth edge in their interval (known result alpha = 1,
f = 2 eps / L); the band points bracket escape windows in the mirror band (stage 8c).

Output: results/uncertainty/summary.json (counts, alpha, resolution scaling, settings, git hash)
and figures/uncertainty_exponent.pdf, figures/uncertainty_resolution.pdf.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics import central                                                     # noqa: E402
from physics.edges import in_plane_moment, strip_angles                         # noqa: E402
from physics.fractal import (effective_length, fit_exponent, outcome_codes,     # noqa: E402
                             resolution_scaling, sample_points, uncertain_counts)
from tests.published.bartz_shaw_2022 import ENERGY_TILDE                        # noqa: E402

SETTINGS = dict(start_radius=64000.0, method="numba")
EPSILONS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]
SCAN_SIZES = [51, 201, 801, 3201, 12801]


def in_plane(line, s_pi, d_pi):
    p1, p2 = strip_angles(line, s_pi * np.pi, d_pi * np.pi)
    return in_plane_moment(p1), in_plane_moment(p2)


Z = np.array([0.0, 0.0, 1.0])
POINTS = {
    # name: (m1, m2, b_lo, b_hi, description)
    "control_zz": (Z, -Z, 4.0, 4.5, "+z/-z: one analytic edge at b_c"),
    "control_anti": (*in_plane("antiparallel", 0.25, 0.0), 4.5, 4.9, "antiparallel line s=pi/4: one smooth edge"),
    "band_A": (*in_plane("mirror", 0.25, -0.03), 1.33, 1.36, "mirror band s=pi/4, d=-0.03 pi"),
    "band_B": (*in_plane("mirror", 0.5, -0.05), 2.225, 2.28, "mirror band s=pi/2, d=-0.05 pi"),
    "band_C": (*in_plane("mirror", 0.25, -0.07), 3.10, 3.30, "mirror band s=pi/4, d=-0.07 pi"),
}


def chunk_job(args):
    energy, m1, m2, b_values, *override = args
    return outcome_codes(energy, m1, m2, b_values, {**SETTINGS, **(override[0] if override else {})})


NUMERICS_VARIANTS = {"rtol 1e-12": dict(rtol=1e-12, atol=1e-14), "r0 128000": dict(start_radius=128000.0)}
NUMERICS_EPS = [1e-4, 1e-5, 1e-6]


def numerics_check(pool, workers, name="band_A", samples=20000, seed=11):
    """Recount the uncertain fraction at one band point with tightened numerics (same samples).

    If the fine structure is dynamical, every variant must give Poisson-consistent counts; if
    integration error decided outcomes near the boundary, tightening would change them.
    """
    m1, m2, b_lo, b_hi, _ = POINTS[name]
    centers, eps, b_all = sample_points(b_lo, b_hi, NUMERICS_EPS, samples, seed)
    chunks = np.array_split(b_all, 4 * workers)
    result = {"point": name, "eps": eps.tolist(), "samples": samples, "variants": {}}
    for label, override in [("production", {})] + list(NUMERICS_VARIANTS.items()):
        codes = "".join(pool.map(chunk_job, [(ENERGY_TILDE, m1, m2, c, override) for c in chunks]))
        rows = uncertain_counts(codes, len(centers), eps)
        result["variants"][label] = {"uncertain": [r[0] for r in rows], "totals": [r[1] for r in rows],
                                     "center_codes": codes[:len(centers)]}
    base_codes = result["variants"]["production"]["center_codes"]
    for label, v in result["variants"].items():
        flips = sum(a != b for a, b in zip(v["center_codes"], base_codes))
        v["center_outcomes_changed"] = flips
        print(f"numerics {label:11s}: uncertain {v['uncertain']} of {v['totals'][0]}; "
              f"center outcomes changed vs production: {flips}")
    for v in result["variants"].values():
        del v["center_codes"]
    return result


def scaling_job(args):
    name, energy, m1, m2, b_lo, b_hi = args
    return name, resolution_scaling(energy, m1, m2, b_lo, b_hi, SCAN_SIZES, SETTINGS)


def refit(summary):
    """Recompute alpha from stored counts with the current fit rule (no new trajectories)."""
    for pt in summary["points"].values():
        alpha, alpha_err, used = fit_exponent(pt["eps"], pt["uncertain"], pt["totals"],
                                              eps_max=(pt["b_hi"] - pt["b_lo"]) / 30)
        pt.update(alpha=alpha, alpha_err=alpha_err, fit_used=used.tolist())
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=50000)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 1))
    p.add_argument("--replot", action="store_true", help="refit and redraw from results/uncertainty/summary.json")
    p.add_argument("--numerics", action="store_true", help="only the tightened-numerics recount (added to summary.json)")
    args = p.parse_args()
    out_dir = ROOT / "results" / "uncertainty"
    if args.replot:
        summary = refit(json.loads((out_dir / "summary.json").read_text()))
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        for name, pt in summary["points"].items():
            print(f"{name:13s} alpha = {pt['alpha']:.3f} +- {pt['alpha_err']:.3f} "
                  f"(fit over eps <= {(pt['b_hi'] - pt['b_lo']) / 30:.1e}, {sum(pt['fit_used'])} points)")
        make_figures(summary)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.numerics:
        summary = json.loads((out_dir / "summary.json").read_text())
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            summary["numerics"] = numerics_check(pool, args.workers)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
        return
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "physics", "analysis/uncertainty.py"], cwd=ROOT,
                                capture_output=True, text=True).stdout)
    summary = {"settings": {**SETTINGS, "energy": ENERGY_TILDE, "epsilons": EPSILONS, "samples": args.samples,
                            "seed": args.seed, "scan_sizes": SCAN_SIZES},
               "git_hash": git, "git_dirty": dirty, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "points": {}}

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        # 1. Uncertainty counts: every point's flat b list, split into chunks across workers.
        for name, (m1, m2, b_lo, b_hi, desc) in POINTS.items():
            t0 = time.time()
            centers, eps, b_all = sample_points(b_lo, b_hi, EPSILONS, args.samples, args.seed)
            chunks = np.array_split(b_all, 4 * args.workers)
            codes = "".join(pool.map(chunk_job, [(ENERGY_TILDE, m1, m2, c) for c in chunks]))
            rows = uncertain_counts(codes, len(centers), eps)
            counts, totals = [r[0] for r in rows], [r[1] for r in rows]
            alpha, alpha_err, used = fit_exponent(eps, counts, totals, eps_max=(b_hi - b_lo) / 30)
            summary["points"][name] = {"description": desc, "b_lo": b_lo, "b_hi": b_hi,
                                       "m1": list(map(float, m1)), "m2": list(map(float, m2)),
                                       "eps": eps.tolist(), "uncertain": counts, "totals": totals,
                                       "timeouts": [r[2] for r in rows], "alpha": alpha, "alpha_err": alpha_err,
                                       "fit_used": used.tolist()}
            print(f"{name:13s} alpha = {alpha:.3f} +- {alpha_err:.3f}   f(eps) = "
                  f"{' '.join(f'{c / t:.1e}' for c, t in zip(counts, totals))}   [{time.time() - t0:.0f} s]", flush=True)
        # 2. Resolution scaling (band points and one control), one point per worker.
        jobs = [(n, ENERGY_TILDE, *POINTS[n][:4]) for n in POINTS if n != "control_zz"]
        for name, rows in pool.map(scaling_job, jobs):
            summary["points"][name]["resolution"] = rows
            print(f"{name:13s} edges vs scan: {[r['n_edges'] for r in rows]}  measure: "
                  f"{[round(r['capture_measure'], 8) if r['capture_measure'] == r['capture_measure'] else None for r in rows]}")

    # 3. Known result for the analytic control: f = 2 eps / L for one edge in an interval of length L.
    zz = summary["points"]["control_zz"]
    L = effective_length(zz["b_lo"], zz["b_hi"], zz["eps"])
    ratios = [c / t / (2 * e / L) for c, t, e in zip(zz["uncertain"], zz["totals"], zz["eps"])]
    print("control_zz: f / (2 eps / L) =", " ".join(f"{r:.3f}" for r in ratios),
          f"(b_c = {central.capture_impact_parameter(ENERGY_TILDE, 1.0):.4f})")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    make_figures(summary)


def make_figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.grid": False, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for k, (name, pt) in enumerate(summary["points"].items()):
        eps = np.array(pt["eps"]); f = np.array(pt["uncertain"]) / np.array(pt["totals"])
        ok = np.array(pt["uncertain"]) > 0
        ax.loglog(eps[ok], f[ok], "o-", color=f"C{k}", mfc="none",
                  label=fr"{name}: $\alpha={pt['alpha']:.2f}\pm{pt['alpha_err']:.2f}$")
    e = np.array(summary["settings"]["epsilons"])
    zz = summary["points"]["control_zz"]
    L = effective_length(zz["b_lo"], zz["b_hi"], zz["eps"])
    ax.loglog(e, 2 * e / L, "k:", lw=1, label=fr"$2\epsilon/L$: one edge, $L={L:.2f}$ (control_zz)")
    ax.set_xlabel(r"$\epsilon/R$"); ax.set_ylabel(r"uncertain fraction $f(\epsilon)$"); ax.legend(fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"uncertainty_exponent.{ext}", dpi=110)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, (name, pt) in enumerate(summary["points"].items()):
        if "resolution" not in pt:
            continue
        db = [r["db"] for r in pt["resolution"]]
        ax1.loglog(db, [max(r["n_edges"], 0.8) for r in pt["resolution"]], "o-", color=f"C{k}", mfc="none", label=name)
        m = np.array([r["capture_measure"] for r in pt["resolution"]])
        ax2.semilogx(db, m / m[-1], "o-", color=f"C{k}", mfc="none", label=name)
    ax1.set_xlabel(r"scan spacing $\Delta b/R$"); ax1.set_ylabel("edges resolved"); ax1.legend(fontsize=10)
    ax2.set_xlabel(r"scan spacing $\Delta b/R$"); ax2.set_ylabel("capture measure / finest")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"uncertainty_resolution.{ext}", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
