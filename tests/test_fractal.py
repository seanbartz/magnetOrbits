"""Stage 8A — uncertainty-exponent machinery (physics/fractal.py).

Checks
------
* known result: one analytic edge (+z/-z, b_c from physics/central.py) in an interval of
  length L gives an uncertain fraction f = 2 eps / L (Poisson-consistent, 4 sigma);
* fit_exponent recovers a known power law from synthetic counts;
* uncertain_counts: layout of the flat outcome string, and timeouts excluded, not counted.
"""

import numpy as np
import pytest

from physics import central
from physics.fractal import effective_length, fit_exponent, outcome_codes, sample_points, uncertain_counts
from tests.published.bartz_shaw_2022 import ENERGY_TILDE

Z = np.array([0.0, 0.0, 1.0])
SETTINGS = dict(start_radius=64000.0, method="numba")


def test_single_analytic_edge_gives_two_eps_over_L():
    b_lo, b_hi, n = 4.0, 4.5, 4000
    assert b_lo < central.capture_impact_parameter(ENERGY_TILDE, 1.0) < b_hi
    centers, eps, b_all = sample_points(b_lo, b_hi, [2e-2, 5e-3], n, seed=7)
    rows = uncertain_counts(outcome_codes(ENERGY_TILDE, Z, -Z, b_all, SETTINGS), n, eps)
    for (count, total, timeouts), e in zip(rows, eps):
        expected = total * 2 * e / effective_length(b_lo, b_hi, eps)
        assert timeouts == 0
        assert abs(count - expected) < 4 * np.sqrt(expected), (count, expected)


def test_fit_exponent_recovers_power_law():
    eps = np.array([1e-2, 1e-3, 1e-4, 1e-5])
    totals = np.full(4, 10**7)
    counts = np.round(totals * 0.3 * eps**0.6).astype(int)
    alpha, alpha_err, used = fit_exponent(eps, counts, totals)
    assert alpha == pytest.approx(0.6, abs=0.01)
    assert used.all() and alpha_err > 0


def test_uncertain_counts_layout_and_timeouts():
    # 3 samples, 1 eps: rows = [center, center - eps, center + eps]
    eps = np.array([1e-3])
    codes = "cec" + "cct" + "cec"     # sample 0 certain; sample 1 (center 'e') differs at -eps; sample 2 times out at -eps
    count, total, timeouts = uncertain_counts(codes, 3, eps)[0]
    assert (count, total, timeouts) == (1, 2, 1)
