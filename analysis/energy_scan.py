"""Stage 6a: collision edge vs. energy, for head-to-tail and averaged over in-plane orientations.

Usage
-----
    python analysis/energy_scan.py run        # one symmetry-reduced 32 x 32 sweep per energy (numba)
    python analysis/energy_scan.py analyze    # checks, summary, figures
    python analysis/energy_scan.py all        # both

Per energy E (ENERGIES, 1e-3 .. 10, log-spaced, incl. the paper energy and the crossovers):
  * results/escan_<k>/ : sweep.py --fundamental --n-phi 32 with the scan_settings(E) rules;
  * head-to-tail edge  = the map point (phi1, phi2) = (0, 0);
  * orientation averages over the 32 x 32 torus from the fundamental domain, each representative
    weighted by the size of its symmetry orbit (exact for the torus grid). The 16 x 16 subgrid
    (even indices, closed under the symmetries) gives the orientation-grid error estimate.

Cross sections (planar, lengths in units of R): sigma_out = 2 <b_out> (robust: b_out is converged,
stage 8); sigma_meas = 2 <capture measure + b_min [b_min collides]>, which uses b -> -b
(phi -> pi - phi, a measure-preserving map of the torus). In the fractal window band the capture
measure is converged only to a few percent (stage 8), so sigma_meas carries that uncertainty.

Scaling rules (scan_settings):
  b_max = max(10, 2 b_c^locked(E))   the locked edge reaches 17 R at E = 1e-3;
  n_scan so that db = (b_max - b_min)/(n_scan - 1) ~ 0.05 (the validated torus-map spacing);
  r0 = max(1000, 64000 E_paper / E)  in-state error ~ (1/(I E r0))^2: hold E r0 at its validated
                                     value (r0 = 64000 at the paper energy); 1000 is the stage-2
                                     validated floor at high energy.

Checks (analyze): +z/-z edge vs analytic on both branches; head-to-tail b_out/b_locked -> 1 as
E -> 0; b_out -> 2 as E -> infinity with (b_out^2 - 4) E finite; at the paper energy the head-to-tail
edge equals the stage-5 value and the map agrees with tests/reference/map_torus32.npz.
Output: results/energy_scan/summary.json, figures/energy_scan_*.pdf.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics import central                                                       # noqa: E402
from physics.edges import capture_profile, completeness_problem, load_map, symmetry_orbit   # noqa: E402
from tests.published.bartz_shaw_2022 import ENERGY_TILDE as E_PAPER               # noqa: E402

ENERGIES = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, E_PAPER, 5e-2, 0.0625, 0.08, 0.125, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_PHI = 32
B_MIN = 0.05
DB_TARGET = 0.05
Z = np.array([0.0, 0.0, 1.0])


def scan_settings(energy):
    """b_max, n_scan and r0 for one energy (see module docstring)."""
    b_max = max(10.0, 2.0 * central.capture_impact_parameter(energy, 2.0))
    n_scan = int(round((b_max - B_MIN) / DB_TARGET)) + 1
    start_radius = max(1000.0, 64000.0 * E_PAPER / energy)
    return {"b_max": float(b_max), "n_scan": n_scan, "start_radius": float(start_radius)}


def run_name(k):
    return f"escan_{k:02d}"


def run(workers=18):
    for k, energy in enumerate(ENERGIES):
        cfg = scan_settings(energy)
        cmd = [sys.executable, "sweep.py", "--name", run_name(k), "--fundamental", "--n-phi", str(N_PHI),
               "--energy", repr(energy), "--b-min", str(B_MIN), "--b-max", repr(cfg["b_max"]),
               "--n-scan", str(cfg["n_scan"]), "--start-radius", repr(cfg["start_radius"]),
               "--method", "numba", "--workers", str(workers)]
        t0 = time.time()
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
        print(f"E = {energy:<9.4g} b_max {cfg['b_max']:6.2f}  n_scan {cfg['n_scan']:4d}  r0 {cfg['start_radius']:9.0f}  "
              f"[{time.time() - t0:.0f} s]", flush=True)


def weighted_stats(records, n):
    """Orientation averages over the n x n torus (and the n/2 x n/2 subgrid) from fundamental reps."""
    tot = {"w": 0.0, "b_out": 0.0, "meas": 0.0, "win": 0.0}
    sub = {"w": 0.0, "b_out": 0.0}
    for (i, j), rec in records.items():
        w = len(symmetry_orbit(i, j, n))
        b_out = rec["outer_edge"]
        meas = rec["capture_measure"] + (B_MIN if rec["scan_codes"][0] == "c" else 0.0)
        tot["w"] += w; tot["b_out"] += w * b_out; tot["meas"] += w * meas
        tot["win"] += w * (rec["n_intervals"] >= 2)
        if i % 2 == 0 and j % 2 == 0:
            sub["w"] += w; sub["b_out"] += w * b_out
    edges = [rec["outer_edge"] for rec in records.values()]
    return {"sigma_out": 2 * tot["b_out"] / tot["w"], "sigma_out_16": 2 * sub["b_out"] / sub["w"],
            "sigma_meas": 2 * tot["meas"] / tot["w"], "window_fraction": tot["win"] / tot["w"],
            "b_out_min": float(min(edges)), "b_out_max": float(max(edges)), "weight_total": tot["w"]}


def antiparallel_z_edge(energy, cfg):
    """+z/-z outer edge with this energy's settings (exact central case, analytic check)."""
    prof = capture_profile(energy, Z, -Z, b_min=B_MIN, b_max=cfg["b_max"], n_scan=cfg["n_scan"],
                           rel_tol=1e-7, start_radius=cfg["start_radius"], method="numba")
    return float(prof["outer_edge"])


def analyze():
    rows = []
    for k, energy in enumerate(ENERGIES):
        cfg = scan_settings(energy)
        data = load_map(ROOT / "results" / run_name(k), expand_symmetry=False)
        problem = completeness_problem(data)
        if problem:
            raise SystemExit(f"{run_name(k)} (E = {energy}) incomplete: {problem}")
        meta = data["meta"]
        stats = weighted_stats(data["records"], N_PHI)
        b_ht = data["records"][(0, 0)]["outer_edge"]
        b_locked = central.capture_impact_parameter(energy, 2.0)
        zz = antiparallel_z_edge(energy, cfg)
        zz_exact = central.capture_impact_parameter(energy, 1.0)
        rows.append({"energy": energy, **cfg, "run": run_name(k), "git_hash": meta["git_hash"],
                     "git_dirty": meta["git_dirty"] or meta.get("git_mixed", False),
                     "b_head_to_tail": b_ht, "b_locked": b_locked, "ratio_locked": b_ht / b_locked,
                     "b_zz": zz, "b_zz_exact": zz_exact, "zz_rel_err": abs(zz / zz_exact - 1), **stats,
                     "sigma_locked": 2 * b_locked})
        r = rows[-1]
        print(f"E {energy:<9.4g} HT b_out {b_ht:7.4f} (locked {b_locked:7.4f}, ratio {r['ratio_locked']:.4f})  "
              f"sigma_out {r['sigma_out']:7.4f} (16x16 {r['sigma_out_16']:7.4f}) sigma_meas {r['sigma_meas']:7.4f}  "
              f"windows {100 * r['window_fraction']:4.1f}%  zz rel err {r['zz_rel_err']:.1e}", flush=True)

    checks = run_checks(rows)
    for name, ok, detail in checks:
        print(f"CHECK {name:28s} {'PASS' if ok else 'FAIL'}  {detail}")
    head, dirty = git_state()
    out = {"energies": ENERGIES, "rows": rows, "checks": [{"name": n, "pass": bool(o), "detail": d} for n, o, d in checks],
           "analysis_git_hash": head, "analysis_git_dirty": dirty, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "results" / "energy_scan").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "energy_scan" / "summary.json").write_text(json.dumps(out, indent=1))
    make_figures(rows)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


def run_checks(rows):
    """Known-result checks on the scan (name, pass, detail)."""
    checks = []
    zz = max(r["zz_rel_err"] for r in rows)
    checks.append(("+z/-z edge vs analytic", zz < 2e-6, f"max rel err {zz:.1e} over {len(rows)} energies (both branches)"))
    low = sorted(rows, key=lambda r: r["energy"])[:4]
    dev = [abs(r["ratio_locked"] - 1) for r in low]
    checks.append(("adiabatic limit E -> 0", dev[0] < dev[-1] and dev[0] < 0.03,
                   "b_out/b_locked - 1 = " + ", ".join(f"{r['ratio_locked'] - 1:+.4f}@{r['energy']:g}" for r in low)))
    high = sorted(rows, key=lambda r: r["energy"])[-4:]
    excess = [(r["b_head_to_tail"] ** 2 - 4) * r["energy"] for r in high]
    gaps = [r["b_head_to_tail"] - 2 for r in high]
    checks.append(("geometric limit E -> inf", gaps[-1] < gaps[0] and gaps[-1] < 0.1 and max(excess) < 3 * min(excess),
                   "b_out - 2 = " + ", ".join(f"{g:.4f}@{r['energy']:g}" for g, r in zip(gaps, high))
                   + "; (b^2-4)E = " + ", ".join(f"{x:.3f}" for x in excess)))
    paper = next(r for r in rows if r["energy"] == E_PAPER)
    checks.append(("paper energy head-to-tail", abs(paper["b_head_to_tail"] / 5.7545902 - 1) < 2e-6,
                   f"{paper['b_head_to_tail']:.7f} vs stage-5 5.7545902"))
    ok, detail = compare_with_torus_map(paper["run"])
    checks.append(("paper energy vs map_torus32", ok, detail))
    dirty = [r["energy"] for r in rows if r["git_dirty"]]
    checks.append(("provenance (clean, unmixed)", not dirty, f"dirty or mixed runs: {dirty}"))
    return checks


def compare_with_torus_map(name):
    """Fundamental-domain points of the paper-energy run vs the stored 32 x 32 production map."""
    ref = np.load(ROOT / "tests" / "reference" / "map_torus32.npz")
    data = load_map(ROOT / "results" / name, expand_symmetry=False)
    same_count, worst = True, 0.0
    for (i, j), rec in data["records"].items():
        same_count &= rec["n_intervals"] == ref["n_intervals"][i, j]
        worst = max(worst, abs(rec["outer_edge"] / ref["outer_edge"][i, j] - 1))
    return same_count and worst < 2e-6, f"interval counts identical: {same_count}; max rel b_out diff {worst:.1e}"


def git_state():
    import sweep
    return sweep.git_state(extra_paths=("analysis/energy_scan.py",))


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.grid": False, "legend.frameon": False})
    E = np.array([r["energy"] for r in rows])
    fine = np.geomspace(E.min(), E.max(), 400)
    locked = np.array([central.capture_impact_parameter(e, 2.0) for e in fine])
    zz = np.array([central.capture_impact_parameter(e, 1.0) for e in fine])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.loglog(fine, locked, "C1-", label=r"locked head-to-tail ($\kappa=2$)")
    ax1.loglog(fine, zz, "C0-", label=r"$+z/-z$ ($\kappa=1$, exact)")
    ax1.axhline(2.0, color="0.5", ls=":", label="contact, $b=2R$")
    ax1.loglog(E, [r["b_head_to_tail"] for r in rows], "ko", mfc="none", label="free rotors, head-to-tail")
    ax1.loglog(E, [r["b_zz"] for r in rows], "C0x", label=r"$+z/-z$ integrator")
    ax1.set_xlabel(r"$\tilde E$"); ax1.set_ylabel(r"collision edge $b_{\rm out}/R$"); ax1.legend(fontsize=11)
    ax2.semilogx(E, [r["ratio_locked"] for r in rows], "ko-", mfc="none", label="head-to-tail")
    ax2.semilogx(E, [r["sigma_out"] / r["sigma_locked"] for r in rows], "C2s-", mfc="none",
                 label=r"orientation average $2\langle b_{\rm out}\rangle$")
    ax2.fill_between(E, [r["b_out_min"] / r["b_locked"] for r in rows], [r["b_out_max"] / r["b_locked"] for r in rows],
                     color="C2", alpha=0.15, label="range over orientations")
    ax2.axhline(1.0, color="C1", lw=1)
    for e_star in (1 / 16, 1 / 8):
        ax2.axvline(e_star, color="0.7", lw=1, ls="--")
    ax2.set_xlabel(r"$\tilde E$"); ax2.set_ylabel(r"$b/b_c^{\rm locked}$"); ax2.legend(fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"energy_scan_edges.{ext}", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogx(E, [100 * r["window_fraction"] for r in rows], "ko-", mfc="none")
    ax.set_xlabel(r"$\tilde E$"); ax.set_ylabel("orientations with escape windows (%)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"energy_scan_windows.{ext}", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("run", "all"):
        run()
    if mode in ("analyze", "all"):
        analyze()
