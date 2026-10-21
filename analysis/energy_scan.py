"""Stage 6a: collision edge vs. energy, for head-to-tail and averaged over in-plane orientations.

Usage
-----
    python analysis/energy_scan.py run        # one symmetry-reduced 32 x 32 sweep per energy (numba)
    python analysis/energy_scan.py analyze    # checks, summary, figures
    python analysis/energy_scan.py all        # both
    python analysis/energy_scan.py save-reference   # summary -> tests/reference/energy_scan.json

Per energy E (ENERGIES, 1e-3 .. 10, log-spaced, incl. the paper energy and the crossovers):
  * results/escan_E<E>/ : sweep.py --fundamental --n-phi 32 with the scan_settings(E) rules (named by
    energy, so adding or removing energies leaves the other runs valid);
  * head-to-tail edge  = the map point (phi1, phi2) = (0, 0);
  * orientation averages over the 32 x 32 torus from the fundamental domain, each representative
    weighted by the size of its symmetry orbit (exact for the torus grid). The 16 x 16 subgrid
    (even indices, closed under the symmetries) gives the orientation-grid error estimate.

Cross sections (planar, lengths in units of R): sigma_out = 2 <b_out> (robust: b_out is converged,
stage 8); sigma_meas = 2 <capture measure + b_min [b_min collides]>, which uses b -> -b
(phi -> pi - phi, a measure-preserving map of the torus). In the fractal window band the capture
measure is converged only to a few percent (stage 8), so sigma_meas carries that uncertainty.

Reference laws (physics/central.py): the locked head-to-tail law with the spin of the co-rotating
moments included, b_c^locked = locked_capture_impact_parameter(E) (the adiabatic limit of free
head-to-tail magnets), and the exact +z/-z law (kappa = 1).

Scaling rules (scan_settings):
  b_max = max(10, 2 b_c^spin-free(E)) b_c^spin-free is the -2/r^3 edge with the spin ignored. This rule
                                     is fixed by the stored runs; the locked edge is at most 1.21 times
                                     b_c^spin-free, so the scan covers it (it reaches 17.5 R at E = 1e-3);
  n_scan so that db = (b_max - b_min)/(n_scan - 1) ~ 0.05 (the validated torus-map spacing);
  r0 = max(1000, 64000 E_paper / E)  in-state error ~ (1/(I E r0))^2: hold E r0 at its validated
                                     value (r0 = 64000 at the paper energy); 1000 is the stage-2
                                     validated floor at high energy.

Before analysis, every run must be complete, have the settings sweep_settings(E) gives (floats to
rel 1e-12, so a run swept on another platform is not rejected over a last-bit difference; a run
made under older scan rules is rejected, and `run` refuses to resume it), and have a finite
outer edge with no contact at the top of the scan (otherwise b_max was too small and b_out would be
underestimated).

Checks (analyze): +z/-z edge vs analytic on both branches; head-to-tail b_out/b_locked -> 1
monotonically as E -> 0, reaching the edge bisection tolerance at the lowest energy; no orientation
collides beyond the locked edge; b_out -> 2 monotonically as E -> infinity with (b_out^2 - 4) E settling to
a constant; at the paper energy the head-to-tail edge equals the stage-5 value
(tests/validated.py) and the map agrees with tests/reference/map_torus32.npz;
provenance: every run clean and unmixed, the analysis tree (runtime code plus ANALYSIS_PATHS)
clean, and the runtime code (runtime_pathspec(): sweep.RUNTIME_PATHS without the analytic oracle
physics/central.py, which no trajectory code imports) identical at every run's commit and at the
analysis commit.
Output: results/energy_scan/summary.json, figures/energy_scan_*.pdf (plus .png previews, gitignored).
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics import central                                                       # noqa: E402
from physics.edges import capture_profile, completeness_problem, load_map, symmetry_orbit   # noqa: E402
from tests.published.bartz_shaw_2022 import ENERGY_TILDE as E_PAPER               # noqa: E402
from tests.validated import STAGE5_HEAD_TO_TAIL_EDGE                              # noqa: E402

from compare_maps import max_rel_diff                                             # noqa: E402

ENERGIES = [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, E_PAPER, 5e-2, 0.0625, 0.08, 0.125, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_PHI = 32
B_MIN = 0.05
DB_TARGET = 0.05
REL_TOL = 1e-6
Z = np.array([0.0, 0.0, 1.0])


def scan_settings(energy):
    """b_max, n_scan and r0 for one energy (see module docstring)."""
    # Fixed by the stored runs: twice the spin-free -2/r^3 edge (covers the locked edge, <= 1.21x).
    b_max = max(10.0, 2.0 * central.capture_impact_parameter(energy, 2.0))
    n_scan = int(round((b_max - B_MIN) / DB_TARGET)) + 1
    start_radius = max(1000.0, 64000.0 * E_PAPER / energy)
    return {"b_max": float(b_max), "n_scan": n_scan, "start_radius": float(start_radius)}


def sweep_settings(energy):
    """The settings dict sweep.py records in meta.json for this energy's run."""
    cfg = scan_settings(energy)
    return {"energy": energy, "b_min": B_MIN, "b_max": cfg["b_max"], "n_scan": cfg["n_scan"],
            "rel_tol": REL_TOL, "start_radius": cfg["start_radius"], "method": "numba"}


def same_settings(have, want):
    """Same keys; floats equal to rel 1e-12 (fractional powers may differ in the last bit across
    platforms), everything else exactly equal."""
    return set(want) == set(have) and all(
        np.isclose(have[k], v, rtol=1e-12, atol=0.0) if isinstance(v, float) else have[k] == v
        for k, v in want.items())


def settings_problem(meta, energy):
    """Why a run's meta.json does not match this scan's settings for this energy, or None.

    Floats are compared to rel 1e-12 (b_max and r0 come from fractional powers, which may differ in
    the last bit between platforms); everything else must be equal.
    """
    want, have = sweep_settings(energy), meta["settings"]
    if same_settings(have, want) and meta["n_phi"] == N_PHI and meta.get("fundamental"):
        return None
    return (f"settings {have} (n_phi {meta['n_phi']}, fundamental {meta.get('fundamental')}) differ from "
            f"this scan's {want}")


def run_name(energy):
    return f"escan_E{energy:.6g}"


def run(workers=min(18, os.cpu_count() or 1)):
    for energy in ENERGIES:
        s = sweep_settings(energy)
        meta_file = ROOT / "results" / run_name(energy) / "meta.json"
        if meta_file.exists():
            problem = settings_problem(json.loads(meta_file.read_text()), energy)
            if problem:
                raise SystemExit(f"results/{run_name(energy)} was made under other scan rules: {problem}. "
                                 f"Move or delete that directory, then run again.")
        cmd = [sys.executable, "sweep.py", "--name", run_name(energy), "--fundamental", "--n-phi", str(N_PHI),
               "--energy", repr(energy), "--b-min", repr(s["b_min"]), "--b-max", repr(s["b_max"]),
               "--n-scan", str(s["n_scan"]), "--rel-tol", repr(s["rel_tol"]),
               "--start-radius", repr(s["start_radius"]), "--method", s["method"], "--workers", str(workers)]
        print(f"E = {energy:<9.4g} b_max {s['b_max']:6.2f}  n_scan {s['n_scan']:4d}  r0 {s['start_radius']:9.0f}  "
              f"-> results/{run_name(energy)}", flush=True)
        t0 = time.time()
        # Output is not captured: sweep.py's progress, and its reason for refusing or failing, stay visible.
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f"  [{time.time() - t0:.0f} s]", flush=True)


def run_problem(data, energy):
    """Why a loaded escan run cannot be used for this energy, or None."""
    problem = completeness_problem(data)
    if problem:
        return f"incomplete: {problem}"
    problem = settings_problem(data["meta"], energy)
    if problem:
        return f"{problem}; move or delete the directory and rerun `energy_scan.py run`"
    no_edge = sorted(k for k, rec in data["records"].items() if rec["outer_edge"] is None)
    at_top = sorted(k for k, rec in data["records"].items() if rec["contact_at_scan_top"])
    if no_edge or at_top:
        return (f"{len(no_edge)} points without an outer edge {no_edge[:5]}, {len(at_top)} still colliding "
                f"at b_max {at_top[:5]} (b_out would be underestimated; raise b_max)")
    return None


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


def _zz_job(energy):
    return antiparallel_z_edge(energy, scan_settings(energy))


def analyze(workers=min(18, os.cpu_count() or 1)):
    # 1. Validate every run before computing anything from it.
    maps = {}
    for energy in ENERGIES:
        data = load_map(ROOT / "results" / run_name(energy), expand_symmetry=False)
        problem = run_problem(data, energy)
        if problem:
            raise SystemExit(f"{run_name(energy)}: {problem}")
        maps[energy] = data
    # 2. The +z/-z analytic check, all energies in parallel.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        zz_edges = dict(zip(ENERGIES, pool.map(_zz_job, ENERGIES)))

    rows = []
    for energy in ENERGIES:
        cfg = scan_settings(energy)
        data = maps[energy]
        meta = data["meta"]
        stats = weighted_stats(data["records"], N_PHI)
        b_ht = data["records"][(0, 0)]["outer_edge"]
        b_locked = central.locked_capture_impact_parameter(energy)
        zz = zz_edges[energy]
        zz_exact = central.capture_impact_parameter(energy, 1.0)
        rows.append({"energy": energy, **cfg, "run": run_name(energy), "git_hash": meta["git_hash"],
                     "git_dirty": meta["git_dirty"] or meta.get("git_mixed", False),
                     "b_head_to_tail": b_ht, "b_locked": b_locked, "ratio_locked": b_ht / b_locked,
                     "b_spin_free": central.capture_impact_parameter(energy, 2.0),
                     "b_zz": zz, "b_zz_exact": zz_exact, "zz_rel_err": abs(zz / zz_exact - 1), **stats,
                     "sigma_locked": 2 * b_locked})
        r = rows[-1]
        print(f"E {energy:<9.4g} HT b_out {b_ht:7.4f} (locked {b_locked:7.4f}, ratio {r['ratio_locked']:.4f})  "
              f"sigma_out {r['sigma_out']:7.4f} (16x16 {r['sigma_out_16']:7.4f}) sigma_meas {r['sigma_meas']:7.4f}  "
              f"windows {100 * r['window_fraction']:4.1f}%  zz rel err {r['zz_rel_err']:.1e}", flush=True)

    head, dirty = git_state()
    checks = run_checks(rows, maps[E_PAPER], head, dirty)
    for name, ok, detail in checks:
        print(f"CHECK {name:28s} {'PASS' if ok else 'FAIL'}  {detail}")
    out = {"energies": ENERGIES, "rows": rows, "checks": [{"name": n, "pass": bool(o), "detail": d} for n, o, d in checks],
           "analysis_git_hash": head, "analysis_git_dirty": dirty, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (ROOT / "results" / "energy_scan").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "energy_scan" / "summary.json").write_text(json.dumps(out, indent=1))
    make_figures(rows)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


def is_strictly_monotone(values):
    diffs = np.diff(values)
    return bool(np.all(diffs > 0) or np.all(diffs < 0))


def run_checks(rows, paper_map, analysis_head, analysis_dirty):
    """Known-result checks on the scan (name, pass, detail)."""
    checks = []
    zz = max(r["zz_rel_err"] for r in rows)
    checks.append(("+z/-z edge vs analytic", zz < 2e-6, f"max rel err {zz:.1e} over {len(rows)} energies (both branches)"))
    # Slow limit: the deviation from the locked law (spin included) shrinks at every step down in
    # energy, and is within the edge bisection tolerance (REL_TOL) at the lowest energy.
    low = sorted(rows, key=lambda r: r["energy"])[:4]
    dev = [abs(r["ratio_locked"] - 1) for r in low]
    checks.append(("adiabatic limit E -> 0", bool(np.all(np.diff(dev) > 0)) and dev[0] < 2 * REL_TOL,
                   "b_out/b_locked - 1 = " + ", ".join(f"{r['ratio_locked'] - 1:+.1e}@{r['energy']:g}" for r in low)))
    # No in-plane orientation collides beyond the locked edge (within the bisection tolerance).
    over = [(r["energy"], r["b_out_max"] / r["b_locked"] - 1) for r in rows if r["b_out_max"] > r["b_locked"] * (1 + 2 * REL_TOL)]
    worst = max(rows, key=lambda r: r["b_out_max"] / r["b_locked"])
    checks.append(("locked edge is the maximum", not over,
                   f"max over orientations / locked: largest {worst['b_out_max'] / worst['b_locked']:.7f} at E = "
                   f"{worst['energy']:g}; above locked: {over}"))
    # Fast limit: b_out - 2 falls at every step up in energy, and (b^2 - 4) E settles monotonically
    # to a constant (spread across the top four energies below 30%).
    high = sorted(rows, key=lambda r: r["energy"])[-4:]
    excess = [(r["b_head_to_tail"] ** 2 - 4) * r["energy"] for r in high]
    gaps = [r["b_head_to_tail"] - 2 for r in high]
    checks.append(("geometric limit E -> inf",
                   bool(np.all(np.diff(gaps) < 0)) and gaps[-1] < 0.02 and is_strictly_monotone(excess)
                   and max(excess) < 1.3 * min(excess),
                   "b_out - 2 = " + ", ".join(f"{g:.4f}@{r['energy']:g}" for g, r in zip(gaps, high))
                   + "; (b^2-4)E = " + ", ".join(f"{x:.3f}" for x in excess)))
    paper = next(r for r in rows if r["energy"] == E_PAPER)
    checks.append(("paper energy head-to-tail", abs(paper["b_head_to_tail"] / STAGE5_HEAD_TO_TAIL_EDGE - 1) < 2e-6,
                   f"{paper['b_head_to_tail']:.7f} vs stage-5 {STAGE5_HEAD_TO_TAIL_EDGE}"))
    ok, detail = compare_with_torus_map(paper_map)
    checks.append(("paper energy vs map_torus32", ok, detail))
    checks.append(("provenance", *provenance(rows, analysis_head, analysis_dirty)))
    return checks


def compare_with_torus_map(data):
    """Fundamental-domain points of the paper-energy run vs the stored 32 x 32 production map.

    Uses compare_maps.max_rel_diff, so an edge missing (nan) in only one of the two maps fails.
    """
    ref = np.load(ROOT / "tests" / "reference" / "map_torus32.npz")
    keys = sorted(data["records"])
    new_edge = [np.nan if data["records"][k]["outer_edge"] is None else data["records"][k]["outer_edge"] for k in keys]
    ref_edge = [ref["outer_edge"][k] for k in keys]
    same_count = all(data["records"][k]["n_intervals"] == ref["n_intervals"][k] for k in keys)
    worst = max_rel_diff(new_edge, ref_edge)
    return same_count and worst < 2e-6, f"interval counts identical: {same_count}; max rel b_out diff {worst:.1e}"


def provenance(rows, analysis_head, analysis_dirty):
    """Every run clean and unmixed, the analysis tree clean, and the runtime code the same at every
    run's commit and at the analysis commit.

    Runs may come from different commits (e.g. only analysis or test files changed in between),
    but not from different versions of the code that computes the trajectories. The analysis itself
    recomputes edges (the +z/-z check) from the working tree, so that must be clean too. A run
    commit missing from the local history (a shallow clone) counts as differing.
    """
    dirty = [r["energy"] for r in rows if r["git_dirty"]]
    hashes = sorted({r["git_hash"] for r in rows})
    changed = [h[:8] for h in hashes
               if subprocess.run(["git", "diff", "--quiet", h, analysis_head, "--", *runtime_pathspec()],
                                 cwd=ROOT, stderr=subprocess.DEVNULL).returncode != 0]
    detail = (f"dirty or mixed runs: {dirty}; analysis tree dirty: {analysis_dirty}; run commits "
              f"{[h[:8] for h in hashes]}; runtime code differs from {analysis_head[:8]} at: {changed}")
    return not dirty and not analysis_dirty and not changed, detail


# physics/central.py is the analytic oracle (power laws, locked law): no trajectory code imports it
# (tests/test_energy_scan.py checks this), so a change there does not invalidate stored runs.
NON_RUNTIME_ORACLES = ("physics/central.py",)


def runtime_pathspec():
    """git pathspec of the code that computes trajectories: sweep.RUNTIME_PATHS minus the oracle."""
    import sweep
    return [*sweep.RUNTIME_PATHS, *(f":(exclude){p}" for p in NON_RUNTIME_ORACLES)]


# Everything the analysis reads besides the runtime code (sweep.RUNTIME_PATHS) and the runs.
ANALYSIS_PATHS = ("analysis/energy_scan.py", "analysis/compare_maps.py", "tests/validated.py",
                  "tests/reference/map_torus32.npz")


def git_state():
    import sweep
    return sweep.git_state(extra_paths=ANALYSIS_PATHS)


def make_figures(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.grid": False, "legend.frameon": False})
    E = np.array([r["energy"] for r in rows])
    fine = np.geomspace(E.min(), E.max(), 400)
    locked = np.array([central.locked_capture_impact_parameter(e) for e in fine])
    spin_free = np.array([central.capture_impact_parameter(e, 2.0) for e in fine])
    zz = np.array([central.capture_impact_parameter(e, 1.0) for e in fine])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1.loglog(fine, locked, "C1-", label="locked head-to-tail (spin included)")
    ax1.loglog(fine, spin_free, "C1:", lw=1, label=r"$-2/r^3$, spin ignored")
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
    ax2.axvline(central.locked_crossover_energy(), color="0.7", lw=1, ls="--")   # locked crossover 0.275
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


def save_reference():
    """Copy the summary to tests/reference/ only if every run and the analysis were clean."""
    summary = json.loads((ROOT / "results" / "energy_scan" / "summary.json").read_text())
    if summary["analysis_git_dirty"] or any(r["git_dirty"] for r in summary["rows"]):
        raise SystemExit("energy-scan summary comes from dirty or mixed runs; rerun from committed code")
    if not all(c["pass"] for c in summary["checks"]):
        raise SystemExit("energy-scan checks did not all pass; not storing a reference")
    out = ROOT / "tests" / "reference" / "energy_scan.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("run", "all"):
        run()
    if mode in ("analyze", "all"):
        analyze()
    if mode == "save-reference":
        save_reference()
