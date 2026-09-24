"""Stage 4 — rotational dynamics of the moments.

Fixed-center checks (translation frozen, separation d along x). Orientation Hessians
derived by expanding U to second order in the tilts at fixed r:

* Head-to-tail (m1 = m2 = x_hat), tilts alpha_i toward y or z:
      U ~ (1/d^3) (-2 + alpha_1^2 + alpha_2^2 + alpha_1 alpha_2),
  eigenvalues 3 (in phase) and 1 (antiphase), each doubly degenerate, so
      omega^2 = 3 / (I d^3)   and   omega^2 = 1 / (I d^3).
* +z / -z, tilts toward r_hat: m1 = (sin a, 0, cos a), m2 = (sin b, 0, -cos b):
      U ~ (1/d^3) (-1 + a^2/2 + b^2/2 - 2 a b),
  eigenvalues -1 (a = b, "scissor": UNSTABLE, growth rate sigma = 1/sqrt(I d^3); started
  from rest a(t) = eps cosh(sigma t)) and +3 (a = -b, omega^2 = 3/(I d^3)).

Free 3D checks: E and J = L + I(w1 + w2) conserved while L alone is not; |m_i| = 1 kept to
the integration tolerance; time-reversal (v, w -> -v, -w with m unchanged) retraces the
trajectory; rigid rotation of the whole initial state rotates the whole solution; global
flip m_i -> -m_i leaves the orbit unchanged (U is bilinear in the moments).

Regression: generic 3D runs saved in tests/reference/stage4_generic.npz.
"""

from pathlib import Path

import numpy as np
import pytest

from physics import observables
from physics.model import SphereParams
from physics.solve import integrate_state, run_trajectory

INERTIA = SphereParams().inertia_tilde
SEPARATION = 3.0
REFERENCE = Path(__file__).parent / "reference" / "stage4_generic.npz"


def fixed_center_state(m1, m2, separation=SEPARATION):
    y = np.zeros(observables.STATE_SIZE)
    y[0:3] = (separation, 0.0, 0.0)
    y[6:9] = np.asarray(m1, float) / np.linalg.norm(m1)
    y[9:12] = np.asarray(m2, float) / np.linalg.norm(m2)
    return y


def measured_frequency(y0, component, expected_omega, n_periods=40):
    """Angular frequency from downward zero crossings of one state component."""
    def crossing(t, y, *args):
        return y[component]
    crossing.direction = -1
    sol = integrate_state(y0, n_periods * 2 * np.pi / expected_omega, fixed_centers=True,
                          rtol=1e-12, atol=1e-16, events=[crossing])
    period = np.mean(np.diff(sol.t_events[0]))
    energy = observables.total_energy(sol.y, INERTIA)
    return 2 * np.pi / period, observables.relative_drift(energy)


@pytest.mark.parametrize("tilt_axis", [1, 2])
@pytest.mark.parametrize("sign,eigenvalue", [(+1, 3.0), (-1, 1.0)])
def test_head_to_tail_normal_modes(tilt_axis, sign, eigenvalue):
    predicted = np.sqrt(eigenvalue / (INERTIA * SEPARATION**3))
    shifts = []
    for amplitude in (1e-4, 2e-4):
        m1 = np.array([1.0, 0.0, 0.0]); m1[tilt_axis] = amplitude
        m2 = np.array([1.0, 0.0, 0.0]); m2[tilt_axis] = sign * amplitude
        omega, energy_drift = measured_frequency(fixed_center_state(m1, m2), 6 + tilt_axis, predicted)
        shifts.append(omega / predicted - 1.0)
        assert energy_drift < 1e-12
    # Harmonic prediction to 1e-8; the residual is the O(amplitude^2) anharmonic shift.
    assert abs(shifts[0]) < 1e-8
    assert shifts[1] / shifts[0] == pytest.approx(4.0, rel=0.01)


def scissor_state(a, b):
    return fixed_center_state((np.sin(a), 0.0, np.cos(a)), (np.sin(b), 0.0, -np.cos(b)))


def test_antiparallel_z_scissor_instability_growth_rate():
    sigma = 1.0 / np.sqrt(INERTIA * SEPARATION**3)
    eps = 1e-10
    times = np.linspace(1.0, 15.0, 8) / sigma
    sol = integrate_state(scissor_state(eps, eps), times[-1], fixed_centers=True,
                          rtol=1e-12, atol=1e-20, dense_output=True)
    tilt = np.arcsin(sol.sol(times)[6])
    np.testing.assert_allclose(tilt / eps, np.cosh(sigma * times), rtol=1e-6)
    # Both moments tilt together (the a = b eigenvector).
    np.testing.assert_allclose(sol.sol(times)[9], sol.sol(times)[6], rtol=1e-6)


def test_antiparallel_z_stable_partner_mode():
    predicted = np.sqrt(3.0 / (INERTIA * SEPARATION**3))
    omega, _ = measured_frequency(scissor_state(1e-4, -1e-4), 6, predicted)
    assert omega / predicted - 1.0 == pytest.approx(0.0, abs=1e-8)


# --- free 3D ------------------------------------------------------------------------------

def generic_cases(n=5, seed=7):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        m1, m2 = rng.normal(size=3), rng.normal(size=3)
        cases.append((rng.uniform(0.01, 0.3), rng.uniform(2.0, 8.0), m1, m2))
    return cases


@pytest.mark.parametrize("energy,b,m1,m2", generic_cases())
def test_generic_3d_conservation(energy, b, m1, m2):
    res = run_trajectory(energy, b, m1, m2, keep_solution=True)
    assert res["energy_drift"] < 1e-8
    assert res["angular_momentum_drift"] < 1e-8
    assert res["moment_norm_error"] < 1e-8
    # Orbital angular momentum is NOT conserved: spin-orbit exchange is O(1) percent or more.
    L = observables.orbital_angular_momentum(res["solution"].y)
    assert observables.relative_drift(L) > 1e-2


def interaction_region_state():
    """A generic state near periapsis, taken from a scattering run."""
    energy, b, m1, m2 = generic_cases()[1]
    res = run_trajectory(energy, b, m1, m2, keep_solution=True)
    sol = res["solution"]
    t_peri = sol.t_events[1][0]
    return sol.sol(t_peri - 20.0)


def time_reversed(y):
    y = y.copy()
    y[3:6] *= -1
    y[12:18] *= -1
    y[18] *= 1   # psi is bookkeeping; not compared
    return y


def test_time_reversal():
    y_start = interaction_region_state()
    forward = integrate_state(y_start, 40.0, rtol=1e-12, atol=1e-14)
    back = integrate_state(time_reversed(forward.y[:, -1]), 40.0, rtol=1e-12, atol=1e-14)
    recovered = time_reversed(back.y[:, -1])
    np.testing.assert_allclose(recovered[:18], y_start[:18], rtol=0, atol=1e-9)


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rotate_state(y, R):
    y = y.copy()
    for k in range(6):   # r, v, m1, m2, w1, w2 are all (pseudo)vectors: same under proper rotations
        y[3 * k:3 * k + 3] = R @ y[3 * k:3 * k + 3]
    return y


def test_rotation_covariance():
    y_start = interaction_region_state()
    R = rotation_matrix([0.3, -1.0, 0.7], 1.1)
    a = integrate_state(y_start, 40.0, rtol=1e-12, atol=1e-14, dense_output=True)
    b = integrate_state(rotate_state(y_start, R), 40.0, rtol=1e-12, atol=1e-14, dense_output=True)
    np.testing.assert_allclose(rotate_state(a.sol(40.0), R)[:18], b.sol(40.0)[:18], rtol=0, atol=1e-9)


def test_global_moment_flip_symmetry():
    energy, b, m1, m2 = generic_cases()[3]
    a = run_trajectory(energy, b, m1, m2)
    c = run_trajectory(energy, b, -m1, -m2)
    np.testing.assert_allclose(c["y_end"][0:6], a["y_end"][0:6], rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(c["y_end"][6:12], -a["y_end"][6:12], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(c["y_end"][12:18], a["y_end"][12:18], rtol=1e-9, atol=1e-14)


# --- regression ---------------------------------------------------------------------------

def compute_generic_reference():
    rows = [run_trajectory(e, b, m1, m2) for e, b, m1, m2 in generic_cases()]
    return {
        "energy": np.array([c[0] for c in generic_cases()]),
        "impact_parameter": np.array([c[1] for c in generic_cases()]),
        "m1": np.array([c[2] for c in generic_cases()]),
        "m2": np.array([c[3] for c in generic_cases()]),
        "outcome": np.array([r["outcome"] for r in rows]),
        "t_end": np.array([r["t_end"] for r in rows]),
        "y_end": np.array([r["y_end"] for r in rows]),
    }


def test_regression_generic_3d():
    ref = np.load(REFERENCE)
    now = compute_generic_reference()
    assert list(now["outcome"]) == list(ref["outcome"])
    # rtol 1e-10 vs 1e-12 changes these by <= 1e-9 relative (measured); 1e-7 leaves margin.
    np.testing.assert_allclose(now["t_end"], ref["t_end"], rtol=1e-7)
    np.testing.assert_allclose(now["y_end"][:, :18], ref["y_end"][:, :18], rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    # Regenerate the regression reference deliberately:  PYTHONPATH=. python tests/test_rotation.py
    import subprocess
    REFERENCE.parent.mkdir(exist_ok=True)
    data = compute_generic_reference()
    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    # Dirty = uncommitted changes in the code that produces the numbers.
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "physics"],
                                capture_output=True, text=True).stdout)
    np.savez(REFERENCE, **data, git_hash=git_hash, git_dirty=dirty,
             settings="run_trajectory defaults: r0=1000, rtol=1e-10, atol=1e-12, beta=0.4")
    print(f"wrote {REFERENCE} (git {git_hash[:8]}, dirty={dirty})")
