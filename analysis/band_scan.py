"""Stage 9: the fractal escape-window band vs. energy (mirror strips and uncertainty exponents).

Usage
-----
    python analysis/band_scan.py strips          # 9a: mirror strips per energy (production + 4x b scan)
    python analysis/band_scan.py exponents       # 9b: uncertainty exponents at band points + controls
    python analysis/band_scan.py analyze         # band measures, checks, summary, figures
    python analysis/band_scan.py all             # the three above
    python analysis/band_scan.py save-reference  # summary + band maps -> tests/reference/

Strips (sweep.py --strip mirror; phi1 = s, phi2 = pi - s + d), per energy E in ENERGIES:
  * results/band_E<E>/        production: the stage-6a settings for E (energy_scan.sweep_settings:
                              b_max = max(10, 2 b_c^locked), db ~ 0.05, r0 = max(1000, 64000 E_paper/E));
  * results/band_E<E>_scan4/  the same strip with a 4x finer b scan; band statistics use this run,
                              since thin windows sit below the production scan (stage 8c);
  * results/band_E0.3_fine/   the closing check: 4x finer d step (pi/2048) and 4x b scan at E = 0.3.
  Along: s in [0, pi) in 32 rows (step pi/32; the other half of the closed line is the flip image).
  Across: step pi/2048 over [-pi/16, pi/64] for E < 0.02 (the band is narrow there), step pi/512
  over [-3pi/8, pi/8] otherwise. Both lattices contain the 32 x 32 torus points (check 5), and the
  wide one contains the stage-8c lattice (check 1). The ranges and the energies come from a
  one-row probe (s = pi/2, 4x scan, 2026-09-25): the band persists down to E = 0.002, moving toward
  the mirror line, and closes between E = 0.25 and 0.28.

Band measures (analyze, from the 4x runs): per row s, the sharp edge (the largest jump of b_out
between neighbouring d) and the d range of window points (n_intervals >= 2); per energy, the
medians over rows, the band's share of the torus (2 x strip window area / 4 pi^2) and the b range
the windows occupy. Window counts are lower bounds (thin windows below the scan are missed).

Exponents (physics.fractal, as in analysis/uncertainty.py): at up to two band points on each of
the rows s = pi/4 and pi/2 of the 4x run (the window point with the most edges, and the window
point at the median d), the b interval spanning that point's windows plus a margin; and two
smooth-edge controls per energy (+z/-z around the analytic b_c; the antiparallel line at s = pi/4
around its outer edge). At the paper energy the stage-8 points band_A and band_B are recomputed
with the stage-8 settings and seed, which must reproduce the stored counts exactly (check 1).

Checks (analyze): (1) paper energy: sharp edges and b_out match the stage-8c strip, stage-8
counts reproduced; (2) controls alpha = 1 at every energy; (3) b_out and the sharp edge unchanged
under the 4x scan; (4) no windows at E = 0.3 on the finer d grid, windows at 0.25; (5) strip points
on the 32 x 32 grid match the stage-6a maps; plus completeness and provenance.
Output: results/band_scan/{exponents,summary}.json, figures/band_scan_*.pdf.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics import central                                                      # noqa: E402
from physics.edges import (capture_profile, completeness_problem, in_plane_moment,   # noqa: E402
                           load_map, strip_angles)
from physics.fractal import (fit_exponent, outcome_codes, resolution_scaling,   # noqa: E402
                             sample_points, uncertain_counts)
from tests.published.bartz_shaw_2022 import ENERGY_TILDE as E_PAPER             # noqa: E402

import energy_scan                                                               # noqa: E402
import uncertainty                                                               # noqa: E402

ENERGIES = [0.002, 0.005, 0.01, 0.02, E_PAPER, 0.0625, 0.125, 0.2, 0.25, 0.3]
CLOSED_ENERGY, LAST_OPEN_ENERGY = 0.3, 0.25
ALONG = [0.0, 1.0, 32]
NARROW = [-0.0625, 0.015625, 161]        # step pi/2048
WIDE = [-0.375, 0.125, 257]              # step pi/512
FINE_CLOSING = {"along_pi": [0.0, 1.0, 8], "across_pi": [-0.3125, -0.0625, 513]}   # step pi/2048
SCAN_FACTOR = 4
EXPONENT_ROWS = (8, 16)                  # s = pi/4, pi/2
SAMPLES, SEED = 50000, 20260925
SCAN_SIZES = [51, 201, 801, 3201, 12801]
RESULTS = ROOT / "results" / "band_scan"
ANALYSIS_PATHS = ("analysis/band_scan.py", "analysis/energy_scan.py", "analysis/uncertainty.py",
                  "tests/reference/uncertainty_summary.json")


# ----------------------------------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------------------------------

def strip_grid(energy, fine=False):
    if fine:
        return {"kind": "strip", "line": "mirror", **FINE_CLOSING}
    return {"kind": "strip", "line": "mirror", "along_pi": ALONG, "across_pi": NARROW if energy < 0.02 else WIDE}


def strip_settings(energy, factor=1):
    s = energy_scan.sweep_settings(energy)
    return {**s, "n_scan": factor * (s["n_scan"] - 1) + 1}


def run_specs():
    """(name, energy, grid, settings) of every strip run."""
    specs = []
    for e in ENERGIES:
        specs.append((f"band_E{e:.6g}", e, strip_grid(e), strip_settings(e)))
        specs.append((f"band_E{e:.6g}_scan4", e, strip_grid(e), strip_settings(e, SCAN_FACTOR)))
    specs.append((f"band_E{CLOSED_ENERGY:.6g}_fine", CLOSED_ENERGY, strip_grid(CLOSED_ENERGY, fine=True),
                  strip_settings(CLOSED_ENERGY, SCAN_FACTOR)))
    return specs


def spec_problem(meta, grid, settings):
    """Why a run's meta.json does not match its spec, or None."""
    if energy_scan.same_settings(meta["settings"], settings) and meta.get("grid") == grid:
        return None
    return f"settings {meta['settings']} / grid {meta.get('grid')} differ from {settings} / {grid}"


def run_strips(workers=min(18, os.cpu_count() or 1)):
    for name, energy, grid, s in run_specs():
        meta_file = ROOT / "results" / name / "meta.json"
        if meta_file.exists():
            problem = spec_problem(json.loads(meta_file.read_text()), grid, s)
            if problem:
                raise SystemExit(f"results/{name} was made under other rules: {problem}. "
                                 "Move or delete that directory, then run again.")
        cmd = [sys.executable, "sweep.py", "--name", name, "--strip", grid["line"],
               "--along-pi", *map(repr, grid["along_pi"][:2]), str(grid["along_pi"][2]),
               "--across-pi", *map(repr, grid["across_pi"][:2]), str(grid["across_pi"][2]),
               "--energy", repr(energy), "--b-min", repr(s["b_min"]), "--b-max", repr(s["b_max"]),
               "--n-scan", str(s["n_scan"]), "--rel-tol", repr(s["rel_tol"]),
               "--start-radius", repr(s["start_radius"]), "--method", s["method"], "--workers", str(workers)]
        print(f"{name}: n_scan {s['n_scan']}, b_max {s['b_max']:.2f}, r0 {s['start_radius']:.0f}, "
              f"grid {grid['along_pi']} x {grid['across_pi']}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, cwd=ROOT, check=True)        # output not captured: progress and errors stay visible
        print(f"  [{time.time() - t0:.0f} s]", flush=True)


def load_run(name, energy, grid, settings):
    """A strip run, validated: complete, matching its spec, every edge resolved inside the scan."""
    data = load_map(ROOT / "results" / name)
    problem = completeness_problem(data) or spec_problem(data["meta"], grid, settings)
    if not problem:
        bad = [k for k, r in data["records"].items() if r["outer_edge"] is None or r["contact_at_scan_top"]]
        problem = f"{len(bad)} points without an outer edge or colliding at b_max: {bad[:5]}" if bad else None
    if problem:
        raise SystemExit(f"{name}: {problem}")
    return data


def load_runs():
    return {name: load_run(name, e, g, s) for name, e, g, s in run_specs()}


# ----------------------------------------------------------------------------------------------
# Band measures
# ----------------------------------------------------------------------------------------------

def escape_windows(rec):
    """Escape windows (lo, hi) inside the contact region: a 'ce' edge followed by an 'ec' edge."""
    edges, kinds = rec["edges"], rec["edge_kinds"]
    return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1) if kinds[k] == "ce" and kinds[k + 1] == "ec"]


def row_measures(outer_edge_row, n_intervals_row, across):
    """Sharp edge (d at the largest b_out jump, midpoint of the two cells), its size, window d range."""
    jumps = np.abs(np.diff(outer_edge_row))
    k = int(np.argmax(jumps))
    win = n_intervals_row >= 2
    return {"sharp_edge": float((across[k] + across[k + 1]) / 2), "sharp_index": k, "jump": float(jumps[k]),
            "window_lo": float(across[win].min()) if win.any() else None,
            "window_hi": float(across[win].max()) if win.any() else None, "n_windows": int(win.sum())}


def grid_steps(grid):
    """(ds, dd) in radians."""
    a_lo, a_hi, n_a = grid["along_pi"]
    c_lo, c_hi, n_c = grid["across_pi"]
    return np.pi * (a_hi - a_lo) / n_a, np.pi * (c_hi - c_lo) / (n_c - 1)


def band_measures(data):
    """Per-row and per-energy band measures of one strip run (d in units of pi)."""
    across = data["across"] / np.pi
    rows = [row_measures(data["outer_edge"][i], data["n_intervals"][i], across) for i in range(len(data["along"]))]
    win = data["n_intervals"] >= 2
    ds, dd = grid_steps(data["grid"])
    windows = [w for (i, j), rec in data["records"].items() if win[i, j] for w in escape_windows(rec)]
    with_win = [r for r in rows if r["n_windows"]]
    med = lambda key, rs: float(np.median([r[key] for r in rs])) if rs else None   # noqa: E731
    return {
        "rows": rows,
        "sharp_edge": med("sharp_edge", rows), "sharp_edge_range": [min(r["sharp_edge"] for r in rows),
                                                                   max(r["sharp_edge"] for r in rows)],
        "window_lo": med("window_lo", with_win), "window_hi": med("window_hi", with_win),
        "n_window_points": int(win.sum()), "rows_with_windows": len(with_win),
        "torus_fraction": float(2 * win.sum() * ds * dd / (4 * np.pi ** 2)),
        "window_b_range": [min(w[0] for w in windows), max(w[1] for w in windows)] if windows else None,
        "windows_at_range_end": bool(win[:, 0].any() or win[:, -1].any()),
        "b_out_min": float(np.min(data["outer_edge"])),
    }


# ----------------------------------------------------------------------------------------------
# Exponents
# ----------------------------------------------------------------------------------------------

def band_points(data):
    """Up to two band points on each EXPONENT_ROWS row: most edges, and median d among window points.

    Each gets the b interval spanning its windows, widened by max(0.02, half their span) on both sides.
    """
    points = {}
    for i in EXPONENT_ROWS:
        js = [j for j in range(len(data["across"])) if data["n_intervals"][i, j] >= 2]
        if not js:
            continue
        most = max(js, key=lambda j: (len(data["records"][(i, j)]["edges"]), -j))
        median = js[len(js) // 2]
        for label, j in (("most_edges", most), ("median_d", median)):
            if any(p["i"] == i and p["j"] == j for p in points.values()):
                continue
            rec = data["records"][(i, j)]
            wins = escape_windows(rec)
            lo, hi = min(w[0] for w in wins), max(w[1] for w in wins)
            pad = max(0.02, 0.5 * (hi - lo))
            points[f"band_s{data['along'][i] / np.pi:.2f}_{label}"] = {
                "i": i, "j": j, "s_pi": float(data["along"][i] / np.pi), "d_pi": float(data["across"][j] / np.pi),
                "phi1": rec["phi1"], "phi2": rec["phi2"], "b_lo": float(max(lo - pad, 0.01)), "b_hi": float(hi + pad),
                "n_edges_4x": len(rec["edges"]), "kind": "band"}
    return points


def control_points(energy, settings):
    """+z/-z around the analytic edge, and the antiparallel line (s = pi/4) around its outer edge."""
    b_zz = central.capture_impact_parameter(energy, 1.0)
    p1, p2 = strip_angles("antiparallel", np.pi / 4, 0.0)
    prof = capture_profile(energy, in_plane_moment(p1), in_plane_moment(p2), b_min=settings["b_min"],
                           b_max=settings["b_max"], n_scan=settings["n_scan"], rel_tol=settings["rel_tol"],
                           start_radius=settings["start_radius"], method=settings["method"])
    b_anti = float(prof["outer_edge"])
    return {"control_zz": {"m": "zz", "b_lo": b_zz - 0.25, "b_hi": b_zz + 0.25, "kind": "control"},
            "control_anti": {"phi1": float(p1), "phi2": float(p2), "b_lo": b_anti - 0.2, "b_hi": b_anti + 0.2,
                             "kind": "control", "n_intervals": prof["n_intervals"]}}


def moments(point):
    """(m1, m2) of a point: explicit moments, +z/-z, or in-plane angles."""
    z = np.array([0.0, 0.0, 1.0])
    if "m1" in point:
        return np.array(point["m1"]), np.array(point["m2"])
    if point.get("m") == "zz":
        return z, -z
    return in_plane_moment(point["phi1"]), in_plane_moment(point["phi2"])


def codes_job(args):
    energy, m1, m2, b_values, settings = args
    return outcome_codes(energy, m1, m2, b_values, settings)


def exponent(pool, workers, energy, point, settings, samples=SAMPLES, seed=SEED):
    """Uncertainty counts and alpha on [b_lo, b_hi] (the analysis/uncertainty.py procedure)."""
    m1, m2 = moments(point)
    b_lo, b_hi = point["b_lo"], point["b_hi"]
    point_eps = [e for e in uncertainty.EPSILONS if e <= (b_hi - b_lo) / 30]
    centers, eps, b_all = sample_points(b_lo, b_hi, point_eps, samples, seed)
    chunks = np.array_split(b_all, 4 * workers)
    codes = "".join(pool.map(codes_job, [(energy, m1, m2, c, settings) for c in chunks]))
    rows = uncertain_counts(codes, len(centers), eps)
    counts, totals = [r[0] for r in rows], [r[1] for r in rows]
    alpha, alpha_err, used = fit_exponent(eps, counts, totals, eps_max=(b_hi - b_lo) / 30)
    return {**point, "eps": eps.tolist(), "uncertain": counts, "totals": totals, "timeouts": [r[2] for r in rows],
            "alpha": alpha, "alpha_err": alpha_err, "fit_used": used.tolist(), "samples": samples, "seed": seed}


def run_exponents(workers=min(18, os.cpu_count() or 1)):
    """Exponents at every energy's band points and controls (resumable: results/band_scan/exponents.json)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "exponents.json"
    head, dirty = git_state()
    store = json.loads(path.read_text()) if path.exists() else {"points": {}}
    store.setdefault("runs", []).append({"git_hash": head, "git_dirty": dirty, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for energy in ENERGIES:
            settings = {"start_radius": energy_scan.scan_settings(energy)["start_radius"], "method": "numba"}
            name = f"band_E{energy:.6g}_scan4"
            data = load_run(name, energy, strip_grid(energy), strip_settings(energy, SCAN_FACTOR))
            points = {**control_points(energy, strip_settings(energy)), **band_points(data)}
            jobs = [(f"{energy:.6g}/{p}", energy, point, settings) for p, point in points.items()]
            if energy == E_PAPER:        # stage-8 points with the stage-8 settings: must reproduce exactly
                for p in ("band_A", "band_B"):
                    m1, m2, b_lo, b_hi, _ = uncertainty.POINTS[p]
                    jobs.append((f"{energy:.6g}/stage8_{p}", energy,
                                 {"m1": list(map(float, m1)), "m2": list(map(float, m2)), "b_lo": b_lo, "b_hi": b_hi,
                                  "kind": "stage8"}, uncertainty.SETTINGS))
            for key, e, point, s in jobs:
                if key in store["points"]:
                    continue
                t0 = time.time()
                res = exponent(pool, workers, e, point, s)
                if point["kind"] == "band":
                    res["resolution"] = resolution_scaling(e, *moments(point), point["b_lo"], point["b_hi"],
                                                           SCAN_SIZES, s)
                store["points"][key] = {**res, "energy": e, "settings": s, "git_hash": head, "git_dirty": dirty}
                path.write_text(json.dumps(store, indent=1))
                print(f"{key:40s} [{point['b_lo']:.4f}, {point['b_hi']:.4f}]  alpha = {res['alpha']:.3f} "
                      f"+- {res['alpha_err']:.3f}  [{time.time() - t0:.0f} s]", flush=True)


# ----------------------------------------------------------------------------------------------
# Analysis and checks
# ----------------------------------------------------------------------------------------------

def torus_index(phi, n=32):
    """Grid index of an angle on the n-point torus grid, or None if it is not on the grid."""
    x = np.mod(phi, 2 * np.pi) / (2 * np.pi / n)
    k = int(round(x)) % n
    return k if abs(x - round(x)) < 1e-9 else None


def compare_with_torus(strip, escan):
    """Strip points that lie on the 32 x 32 grid vs the stage-6a map: (points, interval mismatches, max rel b_out)."""
    n, mism, worst = 0, 0, 0.0
    for (i, j), rec in strip["records"].items():
        a, b = torus_index(rec["phi1"]), torus_index(rec["phi2"])
        if a is None or b is None:
            continue
        n += 1
        mism += rec["n_intervals"] != escan["n_intervals"][a, b]
        worst = max(worst, abs(rec["outer_edge"] / escan["outer_edge"][a, b] - 1))
    return n, mism, worst


def scan_refinement(base, fine):
    """(max rel b_out change, rows whose sharp edge moved, points whose interval count changed)."""
    rel = np.max(np.abs(fine["outer_edge"] / base["outer_edge"] - 1))
    across = base["across"] / np.pi
    moved = sum(row_measures(base["outer_edge"][i], base["n_intervals"][i], across)["sharp_index"]
                != row_measures(fine["outer_edge"][i], fine["n_intervals"][i], across)["sharp_index"]
                for i in range(len(base["along"])))
    return float(rel), int(moved), int(np.sum(base["n_intervals"] != fine["n_intervals"]))


def compare_with_stage8(base):
    """Paper-energy production strip vs the stage-8c strip on their common lattice points."""
    s8 = load_map(ROOT / "results" / "strip_mirror")
    worst, n = 0.0, 0
    for (i, j), rec in base["records"].items():
        i8, j8 = 2 * i, j - 96                       # s step pi/32 vs pi/64; d offset -3/8 vs -3/16
        if not (0 <= j8 < len(s8["across"])):
            continue
        assert abs(s8["along"][i8] - base["along"][i]) < 1e-12 and abs(s8["across"][j8] - base["across"][j]) < 1e-12
        worst = max(worst, abs(rec["outer_edge"] / s8["outer_edge"][i8, j8] - 1)); n += 1
    common = slice(96, 96 + len(s8["across"]))         # sharp edges compared over the common d range
    moved = sum(abs(row_measures(base["outer_edge"][i, common], base["n_intervals"][i, common],
                                 base["across"][common] / np.pi)["sharp_edge"]
                    - row_measures(s8["outer_edge"][2 * i], s8["n_intervals"][2 * i], s8["across"] / np.pi)["sharp_edge"])
                > 1e-12 for i in range(len(base["along"])))
    return n, float(worst), int(moved), s8["meta"]["git_hash"]


def analyze():
    runs = load_runs()
    exps = json.loads((RESULTS / "exponents.json").read_text())["points"]
    per_energy = []
    for e in ENERGIES:
        base, fine = runs[f"band_E{e:.6g}"], runs[f"band_E{e:.6g}_scan4"]
        m = band_measures(fine)
        pts = {k.split("/", 1)[1]: v for k, v in exps.items() if k.split("/", 1)[0] == f"{e:.6g}"}
        band_alphas = [v["alpha"] for k, v in pts.items() if v["kind"] == "band"]
        row = {"energy": e, **{k: v for k, v in m.items() if k != "rows"}, "rows": m["rows"],
               "production_windows": int(np.sum(base["n_intervals"] >= 2)),
               "scan_refinement": scan_refinement(base, fine),
               "exponents": {k: exponent_summary(v) for k, v in pts.items()},
               "alpha_min_band": min(band_alphas) if band_alphas else None,
               "git_hashes": sorted({runs[n]["meta"]["git_hash"] for n in (f"band_E{e:.6g}", f"band_E{e:.6g}_scan4")}),
               "git_dirty": any(runs[n]["meta"]["git_dirty"] or runs[n]["meta"].get("git_mixed", False)
                                for n in (f"band_E{e:.6g}", f"band_E{e:.6g}_scan4"))}
        per_energy.append(row)
        print(f"E {e:<9.4g} sharp edge {m['sharp_edge']:+.4f}pi  windows d in "
              f"[{m['window_lo'] if m['window_lo'] is not None else float('nan'):+.4f}, "
              f"{m['window_hi'] if m['window_hi'] is not None else float('nan'):+.4f}]pi  "
              f"torus fraction {100 * m['torus_fraction']:.2f}%  min alpha {row['alpha_min_band']}", flush=True)
    fine_close = band_measures(runs[f"band_E{CLOSED_ENERGY:.6g}_fine"])
    head, dirty = git_state()
    checks = run_checks(per_energy, runs, exps, fine_close, head, dirty)
    for name, ok, detail in checks:
        print(f"CHECK {name:30s} {'PASS' if ok else 'FAIL'}  {detail}")
    out = {"energies": ENERGIES, "per_energy": per_energy, "closing_fine": {k: v for k, v in fine_close.items() if k != "rows"},
           "fits": fits(per_energy), "checks": [{"name": n, "pass": bool(o), "detail": d} for n, o, d in checks],
           "analysis_git_hash": head, "analysis_git_dirty": dirty, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=1))
    make_figures(out, runs)
    if not all(ok for _, ok, _ in checks):
        raise SystemExit(1)


def exponent_summary(v):
    """The stored fields of one exponent point that the summary keeps."""
    out = {f: v[f] for f in ("alpha", "alpha_err", "b_lo", "b_hi", "kind")}
    out.update({f: v[f] for f in ("s_pi", "d_pi") if f in v})
    if "resolution" in v:
        out["resolution"] = [{f: r[f] for f in ("db", "n_edges", "capture_measure")} for r in v["resolution"]]
    return out


def fits(per_energy):
    """Power law of the sharp edge's distance from the mirror line at low energy (E < 0.05)."""
    low = [r for r in per_energy if r["energy"] < 0.05]
    x, y = np.log([r["energy"] for r in low]), np.log([-r["sharp_edge"] for r in low])
    slope, intercept = np.polyfit(x, y, 1)
    return {"sharp_edge_low_E": {"exponent": float(slope), "prefactor_pi": float(np.exp(intercept)),
                                 "energies": [r["energy"] for r in low]}}


def run_checks(per_energy, runs, exps, fine_close, head, dirty):
    checks = []
    # (1) paper energy vs stage 8.
    n, worst, moved, _ = compare_with_stage8(runs[f"band_E{E_PAPER:.6g}"])
    checks.append(("paper energy vs stage-8c strip", worst < 2e-6 and moved == 0,
                   f"{n} common points, max rel b_out diff {worst:.1e}; rows with a different sharp edge: {moved}"))
    ref = json.loads((ROOT / "tests" / "reference" / "uncertainty_summary.json").read_text())["points"]
    same = {p: exps[f"{E_PAPER:.6g}/stage8_{p}"]["uncertain"] == ref[p]["uncertain"]
            and exps[f"{E_PAPER:.6g}/stage8_{p}"]["totals"] == ref[p]["totals"] for p in ("band_A", "band_B")}
    checks.append(("stage-8 exponents reproduced", all(same.values()),
                   ", ".join(f"{p}: alpha {exps[f'{E_PAPER:.6g}/stage8_{p}']['alpha']:.3f} vs {ref[p]['alpha']:.3f}, "
                             f"counts identical {ok}" for p, ok in same.items())))
    # (2) controls.
    ctrl = [(r["energy"], k, v) for r in per_energy for k, v in r["exponents"].items() if v["kind"] == "control"]
    bad = [(e, k, round(v["alpha"], 3)) for e, k, v in ctrl if not abs(v["alpha"] - 1) <= max(3 * v["alpha_err"], 0.05)]
    checks.append(("controls alpha = 1", not bad and len(ctrl) == 2 * len(per_energy),
                   f"{len(ctrl)} controls, alpha {min(v['alpha'] for *_, v in ctrl):.3f}-{max(v['alpha'] for *_, v in ctrl):.3f}; "
                   f"outside max(3 sigma, 0.05): {bad}"))
    # (3) 4x scan.
    rel = max(r["scan_refinement"][0] for r in per_energy)
    moved = {r["energy"]: r["scan_refinement"][1] for r in per_energy if r["scan_refinement"][1]}
    checks.append(("4x scan: b_out, sharp edge", rel < 2e-6 and not moved,
                   f"max rel b_out change {rel:.1e}; energies with a moved sharp edge: {moved}"))
    # (4) closing.
    open_ = next(r for r in per_energy if r["energy"] == LAST_OPEN_ENERGY)
    closed = next(r for r in per_energy if r["energy"] == CLOSED_ENERGY)
    checks.append(("band closes by E = 0.3", fine_close["n_window_points"] == 0 and closed["n_window_points"] == 0
                   and open_["n_window_points"] > 0,
                   f"window points: {open_['n_window_points']} at E = {LAST_OPEN_ENERGY}, {closed['n_window_points']} at "
                   f"{CLOSED_ENERGY}, {fine_close['n_window_points']} on the pi/2048 grid at {CLOSED_ENERGY}"))
    # (5) torus maps.
    details, ok5 = [], True
    for e in ENERGIES:
        if e not in energy_scan.ENERGIES:
            continue
        escan = load_map(ROOT / "results" / energy_scan.run_name(e))
        n, mism, worst = compare_with_torus(runs[f"band_E{e:.6g}"], escan)
        ok5 &= n > 0 and mism == 0 and worst < 2e-6
        details.append(f"{e:g}: {n} pts, {mism} mismatches, {worst:.0e}")
    checks.append(("strip vs 32x32 maps (6a)", ok5, "; ".join(details)))
    # Range, and provenance.
    at_end = [r["energy"] for r in per_energy if r["windows_at_range_end"]]
    checks.append(("band inside the d range", not at_end, f"energies with windows at a range end: {at_end}"))
    checks.append(("provenance", *provenance(runs, exps, head, dirty)))
    return checks


def provenance(runs, exps, head, dirty):
    import sweep
    run_dirty = [n for n, d in runs.items() if d["meta"]["git_dirty"] or d["meta"].get("git_mixed", False)]
    exp_dirty = sorted({k for k, v in exps.items() if v["git_dirty"]})
    hashes = sorted({d["meta"]["git_hash"] for d in runs.values()} | {v["git_hash"] for v in exps.values()})
    changed = [h[:8] for h in hashes
               if subprocess.run(["git", "diff", "--quiet", h, head, "--", *sweep.RUNTIME_PATHS],
                                 cwd=ROOT, stderr=subprocess.DEVNULL).returncode != 0]
    detail = (f"dirty runs {run_dirty}; dirty exponents {exp_dirty[:3]}{'...' if len(exp_dirty) > 3 else ''}; "
              f"analysis tree dirty {dirty}; commits {[h[:8] for h in hashes]}; runtime code differs at {changed}")
    return not run_dirty and not exp_dirty and not dirty and not changed, detail


def git_state():
    import sweep
    return sweep.git_state(extra_paths=ANALYSIS_PATHS)


# ----------------------------------------------------------------------------------------------
# Figures and reference
# ----------------------------------------------------------------------------------------------

def make_figures(summary, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.grid": False, "legend.frameon": False})
    rows = summary["per_energy"]
    E = np.array([r["energy"] for r in rows])
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    op = [r for r in rows if r["n_window_points"]]
    ax1.loglog(E, [-r["sharp_edge"] for r in rows], "ko-", mfc="none", label="sharp edge")
    ax1.fill_between([r["energy"] for r in op], [-r["window_hi"] for r in op], [-r["window_lo"] for r in op],
                     color="C2", alpha=0.25, label="window points (median over s)")
    f = summary["fits"]["sharp_edge_low_E"]
    xs = np.geomspace(E.min(), 0.05, 50)
    ax1.loglog(xs, f["prefactor_pi"] * xs ** f["exponent"], "k--", lw=1, label=fr"$\propto\tilde E^{{{f['exponent']:.2f}}}$")
    ax1.set_xlabel(r"$\tilde E$"); ax1.set_ylabel(r"distance below the mirror line, $-d/\pi$"); ax1.legend(fontsize=11)
    ax2.semilogx(E, [100 * r["torus_fraction"] for r in rows], "ko-", mfc="none")
    ax2.set_xlabel(r"$\tilde E$"); ax2.set_ylabel("share of orientations with windows (%)")
    for k, kind in (("band", "C3"), ("control", "C0")):
        pts = [(r["energy"], v) for r in rows for v in r["exponents"].values() if v["kind"] == k]
        ax3.errorbar([e for e, _ in pts], [1 - v["alpha"] for _, v in pts], yerr=[v["alpha_err"] for _, v in pts],
                     fmt="o", color=kind, mfc="none", label="band points" if k == "band" else "smooth-edge controls")
    ax3.set_xscale("log"); ax3.axhline(0, color="0.5", lw=1)
    ax3.set_xlabel(r"$\tilde E$"); ax3.set_ylabel(r"uncertainty dimension $D=1-\alpha$"); ax3.legend(fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"band_scan_vs_energy.{ext}", dpi=110)
    plt.close(fig)

    show = [e for e in ENERGIES if e != CLOSED_ENERGY]
    fig, axes = plt.subplots(1, len(show), figsize=(2.4 * len(show), 4.8), sharey=True)
    for ax, e in zip(axes, show):
        d = runs[f"band_E{e:.6g}_scan4"]
        across = d["across"] / np.pi
        ax.imshow(d["n_intervals"] >= 2, origin="lower", aspect="auto", cmap="Greys", interpolation="nearest",
                  extent=(across[0], across[-1], 0, 1))
        ax.axvline(0, color="C0", lw=0.8, ls="--")
        ax.set_title(fr"$\tilde E={e:.3g}$", fontsize=12); ax.set_xlabel(r"$d/\pi$")
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel(r"$s/\pi$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"band_scan_strips.{ext}", dpi=110)
    plt.close(fig)


def save_reference():
    """Summary and compact 4x band maps to tests/reference/, only from clean runs with all checks passing."""
    summary = json.loads((RESULTS / "summary.json").read_text())
    if summary["analysis_git_dirty"] or any(r["git_dirty"] for r in summary["per_energy"]):
        raise SystemExit("band-scan summary comes from dirty or mixed runs; rerun from committed code")
    if not all(c["pass"] for c in summary["checks"]):
        raise SystemExit("band-scan checks did not all pass; not storing a reference")
    runs = load_runs()
    arrays = {}
    for e in ENERGIES:
        d = runs[f"band_E{e:.6g}_scan4"]
        key = f"E{e:.6g}"
        arrays[f"{key}__outer_edge"] = d["outer_edge"]
        arrays[f"{key}__n_intervals"] = d["n_intervals"].astype(np.int8)
        arrays[f"{key}__across"] = d["across"]
        arrays[f"{key}__along"] = d["along"]
        arrays[f"{key}__meta"] = json.dumps({"grid": d["grid"], "settings": d["meta"]["settings"],
                                             "git_hash": d["meta"]["git_hash"], "run": f"band_E{e:.6g}_scan4"})
    np.savez_compressed(ROOT / "tests" / "reference" / "band_maps.npz", **arrays)
    (ROOT / "tests" / "reference" / "band_scan.json").write_text(json.dumps(summary, indent=1))
    print("wrote tests/reference/band_maps.npz and tests/reference/band_scan.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("strips", "all"):
        run_strips()
    if mode in ("exponents", "all"):
        run_exponents()
    if mode in ("analyze", "all"):
        analyze()
    if mode == "save-reference":
        save_reference()
