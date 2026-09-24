"""Stage 6a — collision edge vs energy (analysis/energy_scan.py).

Checks
------
* scaling rules: b_max covers twice the locked edge, db ~ 0.05, r0 = 64000 at the paper energy
  (the validated setting) with E r0 held fixed, and the 1000 floor at high energy;
* orbit-weighted averaging: records sampled on the fundamental domain from a symmetric torus map
  (built without symmetry_orbit) reproduce its exact means (weights sum to n^2; the 16 x 16
  subgrid uses even indices only);
* run validation: wrong settings (a stale or renumbered run), a missing edge, contact at b_max and
  missing points are rejected; the torus-map comparison fails on a one-sided missing edge;
* settings match to rel 1e-12 (a last-bit float difference is accepted); run names are distinct;
* provenance: rules on synthetic rows (dirty run, dirty analysis tree, unknown commit fail); the
  stored scan's runs were made with the current runtime code (skipped without the git history);
* known result: the +z/-z edge at the scan settings matches the analytic b_c on the
  barrier-limited and contact-limited branches;
* regression: head-to-tail edges recomputed at stored energies equal tests/reference/energy_scan.json.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from energy_scan import (B_MIN, ENERGIES, N_PHI, antiparallel_z_edge, compare_with_torus_map,     # noqa: E402
                         provenance, run_name, run_problem, scan_settings, settings_problem, sweep_settings,
                         weighted_stats)
from physics import central                                                                     # noqa: E402
from physics.edges import fundamental_domain, sweep_point                                        # noqa: E402
from tests.published.bartz_shaw_2022 import ENERGY_TILDE as E_PAPER                             # noqa: E402

REFERENCE = ROOT / "tests" / "reference" / "energy_scan.json"


@pytest.mark.parametrize("energy", ENERGIES)
def test_scaling_rules(energy):
    cfg = scan_settings(energy)
    assert cfg["b_max"] >= 2 * central.capture_impact_parameter(energy, 2.0) - 1e-12 and cfg["b_max"] >= 10.0
    db = (cfg["b_max"] - B_MIN) / (cfg["n_scan"] - 1)
    assert 0.049 < db < 0.051
    assert cfg["start_radius"] >= 1000.0
    if cfg["start_radius"] > 1000.0:
        assert cfg["start_radius"] * energy == pytest.approx(64000.0 * E_PAPER, rel=1e-12)
    if energy == E_PAPER:
        assert cfg["start_radius"] == pytest.approx(64000.0, rel=1e-12)


def symmetric_torus_map(n, seed=3):
    """A random n x n map made invariant under the torus symmetries, written out independently of
    physics.edges.symmetry_orbit: (phi1, phi2) -> (phi1 + pi, phi2 + pi) and the swap."""
    g = np.random.default_rng(seed).uniform(2.0, 8.0, size=(n, n))
    h = n // 2
    shifted = np.roll(np.roll(g, -h, axis=0), -h, axis=1)      # g[i + h, j + h]
    return g + g.T + shifted + shifted.T


def test_weighted_stats_are_exact_torus_means():
    n = N_PHI
    full = symmetric_torus_map(n)
    win = (np.floor(full) % 3 == 0)                               # also symmetric
    records = {(i, j): {"outer_edge": full[i, j], "capture_measure": full[i, j] - B_MIN,
                        "scan_codes": "c" + "e" * 3, "n_intervals": 2 if win[i, j] else 1}
               for i, j in fundamental_domain(n)}
    stats = weighted_stats(records, n)
    assert stats["weight_total"] == n * n
    assert stats["sigma_out"] == pytest.approx(2 * full.mean(), rel=1e-14)
    assert stats["sigma_out_16"] == pytest.approx(2 * full[::2, ::2].mean(), rel=1e-14)
    assert stats["sigma_meas"] == pytest.approx(2 * full.mean(), rel=1e-14)      # measure + b_min = value
    assert stats["window_fraction"] == pytest.approx(win.mean(), rel=1e-14)
    assert (stats["b_out_min"], stats["b_out_max"]) == (full.min(), full.max())


def synthetic_run(energy):
    """A loaded-map dict that passes run_problem for this energy."""
    records = {k: {"outer_edge": 3.0, "contact_at_scan_top": False, "unresolved": []} for k in fundamental_domain(N_PHI)}
    meta = {"settings": sweep_settings(energy), "n_phi": N_PHI, "fundamental": True, "grid": {"kind": "torus"}}
    return {"meta": meta, "records": records, "errors": []}


def test_run_problem_rejects_wrong_settings_and_truncated_edges():
    energy = ENERGIES[3]
    assert run_problem(synthetic_run(energy), energy) is None
    assert "differ" in run_problem(synthetic_run(energy), ENERGIES[4])        # renumbered / stale run
    data = synthetic_run(energy)
    data["records"][(0, 0)]["contact_at_scan_top"] = True
    assert "b_max" in run_problem(data, energy)
    data = synthetic_run(energy)
    data["records"][(0, 1)]["outer_edge"] = None
    assert "without an outer edge" in run_problem(data, energy)
    data = synthetic_run(energy)
    del data["records"][(0, 1)]
    assert "incomplete" in run_problem(data, energy)


def test_compare_with_torus_map_fails_on_a_one_sided_missing_edge():
    ref = np.load(REFERENCE.parent / "map_torus32.npz")
    records = {(i, j): {"outer_edge": float(ref["outer_edge"][i, j]), "n_intervals": int(ref["n_intervals"][i, j])}
               for i, j in fundamental_domain(N_PHI)}
    assert compare_with_torus_map({"records": records})[0]
    records[(0, 0)]["outer_edge"] = None
    ok, detail = compare_with_torus_map({"records": records})
    assert not ok and "inf" in detail


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def test_provenance_rules():
    """Synthetic rows at HEAD: independent of which commits the stored scan was made at."""
    head = git_head()
    rows = [{"energy": e, "git_hash": head, "git_dirty": False} for e in ENERGIES[:3]]
    assert provenance(rows, head, False)[0]
    assert not provenance(rows, head, True)[0]                                  # dirty analysis tree
    assert not provenance([dict(rows[0], git_dirty=True)] + rows[1:], head, False)[0]
    assert not provenance(rows + [dict(rows[0], git_hash="0" * 40)], head, False)[0]   # unknown commit


def test_stored_scan_provenance_is_current():
    """The stored scan's runs were made with the current runtime code.

    Fails after any change to sweep.RUNTIME_PATHS: that is the signal to rerun the scan and
    regenerate tests/reference/energy_scan.json. Skipped when the run commits are not in the local
    history (a shallow clone or a source archive).
    """
    rows = json.loads(REFERENCE.read_text())["rows"]
    for h in {r["git_hash"] for r in rows}:
        if subprocess.run(["git", "cat-file", "-e", f"{h}^{{commit}}"], cwd=ROOT,
                          stderr=subprocess.DEVNULL).returncode != 0:
            pytest.skip(f"run commit {h[:8]} not in the local git history")
    ok, detail = provenance(rows, git_head(), False)
    assert ok, f"{detail} -- rerun analysis/energy_scan.py and regenerate the reference"


def test_settings_tolerate_last_bit_float_differences():
    energy = ENERGIES[0]
    meta = synthetic_run(energy)["meta"]
    meta["settings"]["b_max"] = np.nextafter(meta["settings"]["b_max"], np.inf)
    assert settings_problem(meta, energy) is None
    meta["settings"]["b_max"] *= 1 + 1e-9
    assert settings_problem(meta, energy) is not None
    meta = synthetic_run(energy)["meta"]
    meta["settings"]["n_scan"] += 1
    assert settings_problem(meta, energy) is not None


def test_run_names_are_distinct_per_energy():
    assert len({run_name(e) for e in ENERGIES}) == len(ENERGIES)


@pytest.mark.slow
@pytest.mark.parametrize("energy", [2e-3, 1.0])
def test_antiparallel_z_matches_analytic_at_scan_settings(energy):
    edge = antiparallel_z_edge(energy, scan_settings(energy))
    assert edge == pytest.approx(central.capture_impact_parameter(energy, 1.0), rel=2e-6)


@pytest.mark.parametrize("energy", [5e-3, E_PAPER, 2.0])
def test_regression_head_to_tail_against_stored_scan(energy):
    stored = next(r for r in json.loads(REFERENCE.read_text())["rows"] if r["energy"] == energy)
    rec = sweep_point(energy, 0, 0, N_PHI, sweep_settings(energy))
    assert rec["outer_edge"] == pytest.approx(stored["b_head_to_tail"], rel=1e-9)
