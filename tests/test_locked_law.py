"""Locked head-to-tail law with the spin included (physics/central.py, locked_* functions).

Moments held along r_hat co-rotate with the line of centers, so J = (r^2 + 2 I_tilde) theta_dot and
U_eff = J^2 / (2 (r^2 + 2 I_tilde)) - 2/r^3. Checks:
* inertia = 0 reduces to the spin-free -2/r^3 formulas (edge on both branches, deflection);
* the closed-form edge is the true edge of U_eff (brute-force barrier maximum on a fine grid);
* the barrier and contact branches meet at the crossover energy (0.275 for solid spheres);
* known result: free head-to-tail magnets approach the locked law as E -> 0. The stored energy
  scan's head-to-tail edges agree to the bisection tolerance at E <= 0.002, and a live free
  trajectory reproduces the locked deflection (the spin-free law misses it by 4-10%).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from physics import central
from physics.solve import run_trajectory

ROOT = Path(__file__).resolve().parents[1]
E_PAPER = 0.03350081271247681
ENERGIES = [1e-3, 1e-2, 0.1, 0.27, 0.28, 1.0, 10.0]


@pytest.mark.parametrize("energy", ENERGIES)
def test_zero_inertia_is_spin_free_law(energy):
    assert central.locked_capture_impact_parameter(energy, inertia=0.0) == pytest.approx(
        central.capture_impact_parameter(energy, 2.0), rel=1e-12)


@pytest.mark.parametrize("energy, b", [(0.03, 6.0), (0.03, 9.0), (0.5, 3.0), (2.0, 2.5)])
def test_zero_inertia_deflection_is_spin_free_quadrature(energy, b):
    assert central.locked_deflection_angle(energy, b, inertia=0.0) == pytest.approx(
        central.deflection_angle(energy, b, 2.0), abs=1e-11)


@pytest.mark.parametrize("energy", [1e-3, 0.05, 0.2, 0.3, 2.0])
def test_edge_is_barrier_maximum(energy):
    """Just inside the edge U_eff stays below E for all r >= 2; just outside it does not."""
    c = 2.0 * central.SOLID_SPHERE_INERTIA
    r = np.geomspace(2.0, 1e5, 2_000_001)
    edge = central.locked_capture_impact_parameter(energy)
    for factor, collides in ((1 - 1e-7, True), (1 + 1e-7, False)):
        b = edge * factor
        u_max = (b * b * energy / (r * r + c) - 2.0 / r**3).max()
        assert bool(u_max < energy) == collides


def test_branches_meet_at_crossover():
    e_x = central.locked_crossover_energy()
    assert e_x == pytest.approx(0.275, rel=1e-14)
    below = central.locked_capture_impact_parameter(e_x * (1 - 1e-12))
    above = central.locked_capture_impact_parameter(e_x * (1 + 1e-12))
    assert below == pytest.approx(above, rel=1e-9)
    assert central.locked_barrier_radius(e_x) == pytest.approx(2.0, rel=1e-12)


def test_locked_edge_exceeds_spin_free_edge():
    """The spin-up lowers the centrifugal barrier, so the locked edge is larger at every energy."""
    for e in np.geomspace(1e-3, 10, 25):
        assert central.locked_capture_impact_parameter(e) > central.capture_impact_parameter(e, 2.0)


def test_stored_head_to_tail_edges_approach_locked_law():
    """Known result (adiabatic limit) against the stored stage-6a scan: free head-to-tail edges
    agree with the locked law to the edge bisection tolerance (1e-6) at the two lowest energies,
    and the deviation shrinks at every step down in energy."""
    rows = sorted(json.loads((ROOT / "tests" / "reference" / "energy_scan.json").read_text())["rows"],
                  key=lambda r: r["energy"])
    dev = [abs(r["b_head_to_tail"] / central.locked_capture_impact_parameter(r["energy"]) - 1) for r in rows[:5]]
    assert dev[0] < 2e-6 and dev[1] < 2e-6
    assert np.all(np.diff(dev) > 0)


def test_no_orientation_exceeds_locked_edge():
    """Observed in the stage-6a scan: the largest outer edge over in-plane orientations never
    exceeds the locked law (within the bisection tolerance)."""
    rows = json.loads((ROOT / "tests" / "reference" / "energy_scan.json").read_text())["rows"]
    for r in rows:
        assert r["b_out_max"] <= central.locked_capture_impact_parameter(r["energy"]) * (1 + 2e-6)


@pytest.mark.parametrize("energy, b", [(0.002, 16.0), (0.002, 14.5)])
def test_free_head_to_tail_deflection_matches_locked(energy, b):
    """Live: non-spinning head-to-tail magnets at low energy scatter as the locked law predicts
    (to 1e-3 relative at E = 0.002; the spin-free law is off by 4% and 10%)."""
    res = run_trajectory(energy, b, (1, 0, 0), (1, 0, 0), start_radius=64000.0 * E_PAPER / energy,
                         method="numba")
    assert res["outcome"] == "escape"
    locked = central.locked_deflection_angle(energy, b)
    assert res["deflection"] == pytest.approx(locked, rel=1e-3)
    assert abs(res["deflection"] / central.deflection_angle(energy, b, 2.0) - 1) > 0.03
