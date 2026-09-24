"""Outcome classification and timeout bookkeeping in physics/solve.run_trajectory.

Regression tests for review findings on PR #3:
* the default timeout must cover the trip to r_out when r_out > r0 (it was sized from r0
  alone, so runs ended early and reported a non-asymptotic deflection);
* "escape" is declared only when the rigorous criterion fires, never merely because r_out
  was reached, and in stop="escape" mode reaching r_out does not end the run;
* run_single.py accepts a spinning in-state (the asymptotic correction must be off then).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

from physics.solve import run_trajectory

X_HAT = np.array([1.0, 0.0, 0.0])
ROOT = Path(__file__).resolve().parents[1]


def test_default_timeout_covers_large_outgoing_radius():
    res = run_trajectory(0.0335, 6.0, X_HAT, X_HAT, start_radius=250.0, outgoing_radius=4000.0)
    assert res["outcome"] == "escape"
    assert abs(np.linalg.norm(res["y_end"][0:3]) - 4000.0) < 1e-6
    assert np.isfinite(res["deflection_3d"])


def test_small_outgoing_radius_is_not_called_escape():
    # r_out = 4.5 is reached right after periapsis, before r'^2/2 > 2/r^3 can hold.
    res = run_trajectory(0.0335, 6.0, X_HAT, X_HAT, outgoing_radius=4.5)
    assert res["outcome"] == "undecided"
    assert np.isnan(res["deflection"]) and np.isnan(res["deflection_3d"])


def test_escape_mode_ignores_outgoing_radius():
    res = run_trajectory(0.0335, 6.0, X_HAT, X_HAT, outgoing_radius=4.5, stop="escape")
    assert res["outcome"] == "escape"
    assert np.isfinite(res["escape_time"])
    assert res["t_end"] == res["escape_time"]


def test_timeout_is_reported():
    res = run_trajectory(0.0335, 6.0, X_HAT, X_HAT, max_time=100.0)
    assert res["outcome"] == "timeout"
    assert np.isnan(res["deflection_3d"])


def test_run_single_accepts_spinning_in_state():
    out = subprocess.run([sys.executable, "run_single.py", "--energy", "0.03", "--b", "5",
                          "--m1", "1", "0", "0", "--m2", "1", "0", "0", "--w1", "0", "0", "0.01"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "refer to r0" in out.stdout
