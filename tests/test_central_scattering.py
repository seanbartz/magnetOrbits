"""Stage 2 — single trajectories in the exact central case (moments +z / -z).

With m1 = +z, m2 = -z and motion in the xy plane, the torques vanish identically and
U = -1/r^3 (kappa = 1), so the full 19-component integrator must reproduce the central
orbit integral in physics/central.py, which shares no code with it.

Error budget at the default settings (r0 = 1000, rtol = 1e-10, atol = 1e-12):
  * incoming/outgoing legs outside r0: 2 * 3 kappa b / (8 E r0^4) ~ 1e-10 (measured 1.2e-10
    at E = 0.03, b = 5 -- see test_start_radius_convergence_order);
  * integration error: ~1e-10 (rtol plateau).
So 1e-8 is a comfortable absolute tolerance on Theta away from the orbiting singularity.
"""

import numpy as np
import pytest

from physics import central
from physics.solve import run_trajectory

Z = np.array([0.0, 0.0, 1.0])

# (E, b): barrier-limited (E < 1/16) and contact-limited (E > 1/16) regimes, from gentle
# deflection to more than a half turn; all outside the capture edge.
CASES = [(0.01, 8.0), (0.01, 6.8), (0.03, 12.0), (0.03, 5.0), (0.03, 4.6),
         (0.1, 4.0), (0.1, 3.1), (0.3, 3.0), (0.3, 2.45), (1.0, 2.3)]


@pytest.mark.parametrize("energy,b", CASES)
def test_deflection_matches_quadrature(energy, b):
    assert b > central.capture_impact_parameter(energy, 1.0)
    res = run_trajectory(energy, b, Z, -Z)
    expected = central.deflection_angle(energy, b, 1.0)
    assert res["outcome"] == "escape"
    assert res["deflection"] == pytest.approx(expected, abs=1e-8), (res["deflection"], expected)
    # Turning point: located by the periapsis event to the event-root tolerance.
    assert res["r_min"] == pytest.approx(central.turning_point(energy, b, 1.0), rel=1e-9)


@pytest.mark.parametrize("energy,b", CASES[::3])
def test_moments_stay_exactly_fixed_and_invariants(energy, b):
    res = run_trajectory(energy, b, Z, -Z)
    y = res["y_end"]
    assert np.all(y[6:9] == Z) and np.all(y[9:12] == -Z)      # bit-for-bit
    assert np.all(y[12:18] == 0.0)                             # no spin ever generated
    assert res["energy_drift"] < 1e-9
    assert res["angular_momentum_drift"] < 1e-9
    assert res["moment_norm_error"] == 0.0


def test_repulsive_parallel_z():
    """m1 = m2 = +z gives U = +1/r^3 (kappa = -1); same quadrature, repulsive branch."""
    for energy, b in [(0.03, 2.0), (0.3, 2.2), (1.0, 3.0)]:
        res = run_trajectory(energy, b, Z, Z)
        expected = central.deflection_angle(energy, b, -1.0)
        assert expected > 0
        assert res["deflection"] == pytest.approx(expected, abs=1e-8)


def test_small_angle_limit():
    """Theta * E b^3 / (-2 kappa) -> 1 with a correction that shrinks ~ b^-3."""
    energy = 0.1
    ratios = []
    for b in (20.0, 40.0, 80.0):
        res = run_trajectory(energy, b, Z, -Z, rtol=1e-12, atol=1e-14, start_radius=4000.0)
        ratios.append(res["deflection"] / central.small_angle_deflection(energy, b, 1.0))
    corrections = np.abs(np.array(ratios) - 1.0)
    assert corrections[-1] < 1e-4
    # Each doubling of b should shrink the correction by ~8 (next order is O(Theta)).
    assert np.all(corrections[1:] / corrections[:-1] < 1 / 6)


def test_start_radius_convergence_order():
    """The residual comes from the legs outside r0 and should fall as r0^-4."""
    energy, b = 0.03, 5.0
    expected = central.deflection_angle(energy, b, 1.0)
    errors = []
    for r0 in (125.0, 250.0, 500.0):
        res = run_trajectory(energy, b, Z, -Z, rtol=1e-13, atol=1e-15, start_radius=r0)
        errors.append(abs(res["deflection"] - expected))
    order = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert np.all((order > 3.7) & (order < 4.3)), order


def test_rtol_convergence():
    """Tightening rtol 1e-10 -> 1e-12 moves Theta by less than the r0 error budget."""
    energy, b = 0.03, 5.0
    values = [run_trajectory(energy, b, Z, -Z, rtol=rtol, atol=rtol * 1e-2)["deflection"]
              for rtol in (1e-10, 1e-11, 1e-12)]
    assert abs(values[0] - values[2]) < 5e-10
    assert abs(values[1] - values[2]) < 5e-10
