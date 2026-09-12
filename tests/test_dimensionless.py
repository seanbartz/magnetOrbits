"""Step 1: only (Etil, btil, phi1, phi2) can appear.

Two *physically different* dipole pairs are given initial conditions with the same
dimensionless parameters.  Their full dimensional trajectories, integrated in lab
Cartesian coordinates with explicit mu0, m, I and magnetic moment, must map onto the
*same* dimensionless trajectory -- and onto the one produced by `magdip.dynamics`.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from magdip import dynamics as dyn
from magdip import scatter
from magdip.dimensional import DipolePair

# neodymium-sphere-ish, and something deliberately unphysical, with wildly
# different masses, inertias, moments and even a different mu0
PAIRS = [
    DipolePair(mass=0.0312, inertia=2.7e-6, moment=0.44),
    DipolePair(mass=7.5, inertia=1.3e-2, moment=61.0),
    DipolePair(mass=1.0, inertia=1.0, moment=1.0, mu0=2.0 * np.pi),
]

PARAMS = [(0.2, 3.0, 0.7, -1.9), (0.05, 5.0, 2.4, 0.3), (1.5, 1.2, -0.4, 2.8)]


@pytest.mark.parametrize("Etil,btil,phi1,phi2", PARAMS)
def test_dimensional_trajectories_collapse_onto_one_dimensionless_trajectory(
        Etil, btil, phi1, phi2):
    rho0, tau_end = 40.0, 60.0

    y0 = scatter.initial_state(Etil, btil, phi1, phi2, rho0=rho0)
    ref = solve_ivp(dyn.rhs, (0.0, tau_end), y0, method="DOP853",
                    rtol=1e-12, atol=1e-13).y[:, -1]

    for pair in PAIRS:
        state0 = pair.from_dimensionless(y0)
        # the dimensional set-up must reproduce the intended dimensionless parameters
        assert pair.energy(state0) / pair.energy_scale == pytest.approx(
            dyn.energy(y0), rel=1e-12)
        assert pair.total_angular_momentum(state0) / pair.momentum_scale == \
            pytest.approx(dyn.total_angular_momentum(y0), rel=1e-12)

        t_end = tau_end * pair.time_scale
        sol = solve_ivp(pair.rhs, (0.0, t_end), state0, method="DOP853",
                        rtol=1e-13, atol=1e-16 * np.maximum(np.abs(state0), 1e-3))
        got = pair.to_dimensionless(sol.y[:, -1])
        assert np.allclose(got, ref, rtol=2e-7, atol=2e-7), pair.__dict__


def test_energy_parameter_scales_as_E_times_I_to_the_three_halves():
    """Etil = E a^3 / C with a = sqrt(2I/m), so Etil ~ E I^(3/2) m^(-3/2)."""
    base = DipolePair(mass=0.03, inertia=2e-6, moment=0.4)
    scaled = DipolePair(mass=0.03, inertia=2e-6 * 4.0, moment=0.4)
    assert scaled.length_scale / base.length_scale == pytest.approx(2.0)
    assert base.energy_scale / scaled.energy_scale == pytest.approx(8.0)


def test_length_scale_is_radius_of_gyration():
    pair = DipolePair(mass=0.03, inertia=2e-6, moment=0.4)
    assert pair.length_scale == pytest.approx(np.sqrt(2.0 * pair.inertia / pair.mass))
    assert pair.C == pytest.approx(pair.mu0 * pair.moment ** 2 / (2 * np.pi))


def test_dimensionless_roundtrip():
    pair = DipolePair(mass=0.31, inertia=4.2e-5, moment=1.7)
    rng = np.random.default_rng(7)
    for _ in range(20):
        y = np.concatenate([[rng.uniform(0.5, 8.0)], rng.uniform(-np.pi, np.pi, 3),
                            rng.uniform(-1, 1, 4)])
        back = pair.to_dimensionless(pair.from_dimensionless(y))
        assert np.allclose(back, y, rtol=1e-12, atol=1e-12)
