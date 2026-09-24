"""Where the initial velocity goes.

The asymptotic in-state carries three kinds of velocity:

1. the centre-of-mass velocity      -- irrelevant (Galilean invariance),
2. the relative speed ``v``         -- absorbed entirely into ``Etil = v**2/2``,
3. the initial spins ``s1, s2``     -- *not* absorbed: genuine extra parameters that
                                       every scan so far has held at zero.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from magdip import dynamics as dyn
from magdip import scatter
from magdip.dimensional import DipolePair


def _params(pair, v, b):
    """Dimensionless (Etil, btil) for a physical relative speed and impact parameter."""
    E = 0.25 * pair.mass * v ** 2          # = mu v^2 / 2 with mu = m/2
    return E / pair.energy_scale, b / pair.length_scale


def test_relative_speed_enters_only_through_Etil():
    """Etil = v^2 I^(3/2) / (sqrt(2) m^(1/2) C): a change in v is exactly a change in I.

    Two dimensionally different experiments -- one three times faster than the other,
    with the moment of inertia compensating by 3^(-4/3) -- are integrated separately in
    full dimensional lab coordinates and must give the same dimensionless trajectory.
    """
    m, moment = 0.031, 0.44
    I_a, v_a, b_a = 2.7e-6, 0.85, 0.0396   # btil ~ 3, comfortably a fly-by
    fast = 3.0
    I_b = I_a * fast ** (-4.0 / 3.0)
    pair_a = DipolePair(mass=m, inertia=I_a, moment=moment)
    pair_b = DipolePair(mass=m, inertia=I_b, moment=moment)
    v_b = fast * v_a
    b_b = b_a * pair_b.length_scale / pair_a.length_scale

    Ea, ba = _params(pair_a, v_a, b_a)
    Eb, bb = _params(pair_b, v_b, b_b)
    assert Eb == pytest.approx(Ea, rel=1e-12), "the I compensation is wrong"
    assert bb == pytest.approx(ba, rel=1e-12)

    tau_end = 60.0
    y0 = scatter.initial_state(Ea, ba, 0.7, -1.9, rho0=40.0)
    sol0 = solve_ivp(dyn.rhs, (0.0, tau_end), y0, method="DOP853",
                     rtol=1e-12, atol=1e-13)
    ref = sol0.y[:, -1]
    # the comparison is meaningless past a collapse, where trajectories separate
    # exponentially no matter how good the integrator is
    assert sol0.y[dyn.IRHO].min() > 0.5, "pick parameters that do not collapse"
    for pair in (pair_a, pair_b):
        st = pair.from_dimensionless(y0)
        sol = solve_ivp(pair.rhs, (0.0, tau_end * pair.time_scale), st,
                        method="DOP853", rtol=1e-13,
                        atol=1e-16 * np.maximum(np.abs(st), 1e-3))
        assert np.allclose(pair.to_dimensionless(sol.y[:, -1]), ref,
                           rtol=2e-7, atol=2e-7)


def test_doubling_speed_is_a_known_shift_in_Etil():
    """At fixed material properties, Etil ~ v^2 -- nothing else changes."""
    pair = DipolePair(mass=0.031, inertia=2.7e-6, moment=0.44)
    E1, b1 = _params(pair, 1.0, 0.004)
    E2, b2 = _params(pair, 2.0, 0.004)
    assert E2 / E1 == pytest.approx(4.0, rel=1e-12)
    assert b2 == pytest.approx(b1, rel=1e-12), "btil does not depend on speed"


def test_centre_of_mass_velocity_is_irrelevant():
    """Galilean invariance: boosting the pair cannot change the relative dynamics."""
    pair = DipolePair(mass=0.031, inertia=2.7e-6, moment=0.44)
    y0 = scatter.initial_state(0.2, 3.0, 0.7, -1.9, rho0=40.0)
    st = pair.from_dimensionless(y0)
    boosted = st.copy()
    V = np.array([0.37, -0.94])
    boosted[6:8] += V
    boosted[8:10] += V
    boosted[0:2] += np.array([0.011, -0.006])
    boosted[2:4] += np.array([0.011, -0.006])

    t_end = 60.0 * pair.time_scale
    out = []
    for state in (st, boosted):
        sol = solve_ivp(pair.rhs, (0.0, t_end), state, method="DOP853", rtol=1e-13,
                        atol=1e-16 * np.maximum(np.abs(state), 1e-3))
        out.append(pair.to_dimensionless(sol.y[:, -1]))
    assert np.allclose(out[0], out[1], rtol=1e-7, atol=1e-7)


def test_initial_spin_is_a_genuine_extra_parameter():
    """At the *same* total Etil, btil and orientations, spin changes the outcome.

    This is the part of the in-state that the (Etil, btil, phi1, phi2) family does not
    cover: it assumes a non-rotating beam.
    """
    E_total, b = 0.3, 1.9
    base = scatter.run_trajectory(E_total, b, 1.9, 1.3)
    y0 = scatter.initial_state_total_energy(E_total, b, 1.9, 1.3, s1=0.25, s2=-0.15)
    spun = scatter.run_trajectory(E_total, b, 1.9, 1.3, y0=y0)
    assert dyn.energy(y0) == pytest.approx(dyn.energy(base["sol"].y[:, 0]), abs=1e-9)
    assert dyn.total_angular_momentum(y0) != pytest.approx(base["J0"])
    assert (spun["outcome_name"] != base["outcome_name"]
            or abs(spun["chi"] - base["chi"]) > 1e-3)


def test_total_energy_split_is_consistent():
    y0 = scatter.initial_state_total_energy(0.5, 2.0, 0.3, -0.4, rho0=400.0,
                                            s1=0.4, s2=-0.2)
    assert dyn.energy(y0) == pytest.approx(0.5, abs=1e-7)
    with pytest.raises(ValueError):
        scatter.initial_state_total_energy(0.05, 2.0, s1=1.0)


def test_zero_spin_default_is_unchanged():
    """Backwards compatibility: the default in-state is exactly the old one."""
    y = scatter.initial_state(0.2, 3.0, 0.7, -1.9)
    assert y[dyn.IS1] == 0.0 and y[dyn.IS2] == 0.0
    assert dyn.total_angular_momentum(y) == pytest.approx(y[dyn.IL])


def test_orientation_is_an_asymptotic_label_only_for_a_non_rotating_in_state():
    """With s != 0 the starting angle is a phase referred to rho0, not an asymptote.

    A non-rotating dipole's orientation converges as rho0 -> inf (the residual torque
    integrates to O(1/rho)), so chi(rho0) converges too.  A spinning one keeps turning
    on the way in, so the *same* phi_0 at a different rho0 is a physically different
    in-state.  Any scan over (phi1_0, phi2_0) at non-zero spin is therefore a scan over
    phases at that particular rho0, and must be averaged accordingly.
    """
    E, b, p1, p2 = 0.3, 3.0, 1.9, 1.3
    still = [scatter.run_trajectory(E, b, p1, p2, rho0=r)["chi"]
             for r in (200.0, 400.0, 800.0)]
    spun = [scatter.run_trajectory(E, b, p1, p2, rho0=r, s1=0.1, s2=0.1)["chi"]
            for r in (200.0, 400.0, 800.0)]
    assert max(still) - min(still) < 1e-3, still
    assert max(spun) - min(spun) > 0.1, spun


def test_non_rotating_scans_are_converged_in_rho0():
    """The control that makes the existing results trustworthy."""
    E, p1, p2 = 0.3, 1.9, 1.3
    b = np.linspace(0.2, 5.0, 400)
    edges = []
    for rho0 in (200.0, 400.0):
        r = scatter.run_batch(E, b, p1, p2, rho0=rho0, regularized=True, rtol=1e-11)
        assert r["collapsed"].any()
        edges.append(b[np.max(np.where(r["collapsed"])[0])])
    assert abs(edges[0] - edges[1]) < 0.02, edges
