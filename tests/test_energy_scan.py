"""Stage 6a — collision edge vs energy (analysis/energy_scan.py).

Checks
------
* scaling rules: b_max covers twice the locked edge, db ~ 0.05, r0 = 64000 at the paper energy
  (the validated setting) with E r0 held fixed, and the 1000 floor at high energy;
* orbit-weighted averaging: synthetic fundamental-domain records reproduce exact torus means
  (weights sum to n^2; the 16 x 16 subgrid uses even indices only);
* known result: the +z/-z edge at the scan settings matches the analytic b_c on the
  barrier-limited and contact-limited branches;
* regression: head-to-tail edges recomputed at stored energies equal tests/reference/energy_scan.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from energy_scan import B_MIN, ENERGIES, N_PHI, antiparallel_z_edge, scan_settings, weighted_stats  # noqa: E402
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


def test_weighted_stats_are_exact_torus_means():
    n = N_PHI
    reps = fundamental_domain(n)
    def rec(i, j):
        value = 3.0 + (i + j) % 5          # symmetric under the torus symmetries' index maps? not needed
        return {"outer_edge": value, "capture_measure": value - B_MIN, "scan_codes": "c" + "e" * 3,
                "n_intervals": 2 if (i + j) % 7 == 0 else 1}
    # Assign each orbit the value of its representative, as a symmetric map would have.
    records = {r: rec(*r) for r in reps}
    stats = weighted_stats(records, n)
    assert stats["weight_total"] == n * n
    from physics.edges import symmetry_orbit
    full = np.zeros((n, n)); win = np.zeros((n, n))
    for r in reps:
        for a, b in symmetry_orbit(*r, n):
            full[a, b] = records[r]["outer_edge"]; win[a, b] = records[r]["n_intervals"] >= 2
    assert stats["sigma_out"] == pytest.approx(2 * full.mean(), rel=1e-14)
    assert stats["sigma_out_16"] == pytest.approx(2 * full[::2, ::2].mean(), rel=1e-14)
    assert stats["sigma_meas"] == pytest.approx(2 * full.mean(), rel=1e-14)      # measure + b_min = value
    assert stats["window_fraction"] == pytest.approx(win.mean(), rel=1e-14)


@pytest.mark.slow
@pytest.mark.parametrize("energy", [2e-3, 1.0])
def test_antiparallel_z_matches_analytic_at_scan_settings(energy):
    edge = antiparallel_z_edge(energy, scan_settings(energy))
    assert edge == pytest.approx(central.capture_impact_parameter(energy, 1.0), rel=2e-6)


@pytest.mark.parametrize("energy", [5e-3, E_PAPER, 2.0])
def test_regression_head_to_tail_against_stored_scan(energy):
    stored = next(r for r in json.loads(REFERENCE.read_text())["rows"] if r["energy"] == energy)
    cfg = scan_settings(energy)
    settings = {"b_min": B_MIN, "b_max": cfg["b_max"], "n_scan": cfg["n_scan"], "rel_tol": 1e-6,
                "start_radius": cfg["start_radius"], "method": "numba"}
    rec = sweep_point(energy, 0, 0, N_PHI, settings)
    assert rec["outer_edge"] == pytest.approx(stored["b_head_to_tail"], rel=1e-9)
