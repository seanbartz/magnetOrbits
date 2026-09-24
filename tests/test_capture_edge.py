"""Stage 3 — the capture (collision) edge in the exact central case (moments +z / -z).

Checks
------
* Bisection edge b_c matches both analytic branches (physics/central.py),
      b_c = (27/4)^(1/6) E^(-1/3)  (E < 1/16, barrier-limited)
      b_c = sqrt(4 + 1/(2E))       (E > 1/16, contact-limited),
  at energies spanning the crossover E = 1/16.
* The analytic branches are continuous at the crossover (b_c = 2 sqrt 3).
* Grazing contacts just inside the contact-limited edge are detected (via the periapsis
  check when the r = 2 sign change falls inside one step).
* Orbiting: Theta diverges as ln(b - b_c) with coefficient exactly 1 (barrier-limited),
  derived in physics/central.py; numerical Theta matches quadrature there, with the
  expected 1/(b - b_c) amplification of a fixed effective error in b.
"""

import numpy as np
import pytest

from physics import central
from physics.capture import find_capture_edge
from physics.solve import run_trajectory

Z = np.array([0.0, 0.0, 1.0])
KAPPA = 1.0


def test_branches_are_continuous():
    e_star = central.crossover_energy(KAPPA)
    assert e_star == pytest.approx(1.0 / 16.0, rel=1e-15)
    below = central.capture_impact_parameter(e_star * (1 - 1e-12), KAPPA)
    above = central.capture_impact_parameter(e_star * (1 + 1e-12), KAPPA)
    assert below == pytest.approx(2 * np.sqrt(3), rel=1e-10)
    assert above == pytest.approx(2 * np.sqrt(3), rel=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("energy", [0.003, 0.01, 0.03, 0.06, 0.065, 0.1, 0.3, 1.0])
def test_bisected_edge_matches_analytic(energy):
    expected = central.capture_impact_parameter(energy, KAPPA)
    found = find_capture_edge(energy, Z, -Z, 0.9 * expected, 1.1 * expected, rel_tol=1e-9)
    assert found["b_contact"] < found["b_escape"]
    # Bracket half-width is 5e-10 relative; the outcome itself is decided to ~1e-10.
    assert found["edge"] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("energy", [0.1, 0.3, 1.0])
def test_grazing_contact_detected(energy):
    edge = central.capture_impact_parameter(energy, KAPPA)
    inside = run_trajectory(energy, edge * (1 - 1e-7), Z, -Z, stop="escape")
    outside = run_trajectory(energy, edge * (1 + 1e-7), Z, -Z, stop="escape")
    assert inside["outcome"] == "contact"
    assert inside["r_min"] < 2.0
    assert outside["outcome"] == "escape"
    assert outside["r_min"] > 2.0


def test_grazing_uses_periapsis_path():
    """At 1e-7 inside the edge the dip below r = 2 lasts less than one step."""
    energy = 0.3
    edge = central.capture_impact_parameter(energy, KAPPA)
    res = run_trajectory(energy, edge * (1 - 1e-7), Z, -Z, stop="escape")
    assert res["contact_via"] == "periapsis"


def test_orbiting_log_divergence():
    energy = 0.01                       # barrier-limited: E < 1/16
    edge = central.capture_impact_parameter(energy, KAPPA)
    offsets = 10.0 ** -np.arange(3, 7)
    numeric = np.array([run_trajectory(energy, edge + d, Z, -Z)["deflection"] for d in offsets])
    quad = np.array([central.deflection_angle(energy, edge + d, KAPPA) for d in offsets])

    slopes_numeric = np.diff(numeric) / np.diff(np.log(offsets))
    slopes_quad = np.diff(quad) / np.diff(np.log(offsets))
    assert np.all(np.abs(slopes_quad - 1.0) < 5e-3)
    assert np.all(np.abs(slopes_quad[-1] - 1.0) < 1e-4)
    assert np.all(np.abs(slopes_numeric - 1.0) < 5e-3)
    # Error model: dTheta/db = 1/(b - b_c), so a fixed effective error db in the in-state/
    # integration gives |Theta_num - Theta_quad| = db / (b - b_c). Check db is small AND
    # constant across three decades (measured ~3.5e-11, i.e. 5e-12 relative to b).
    effective_db = np.abs(numeric - quad) * offsets
    assert np.all(effective_db < 1e-10)
    assert effective_db.max() / effective_db.min() < 1.5
