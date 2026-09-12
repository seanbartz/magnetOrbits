"""The rotational model: one rigid degree of freedom per magnet, s = I phi_dot.

All angular momentum of a magnet in this model comes from the angular velocity of its
moment's orientation.  The electron-spin angular momentum that a real ferromagnet also
carries (Einstein--de Haas / Barnett) is deliberately excluded; these tests pin that
choice down and size what it costs.
"""
import numpy as np
import pytest

from magdip import dynamics as dyn
from magdip import locked, scatter
from magdip.dimensional import GAMMA_E, DipolePair

# the 5 mm NdFeB spheres of arXiv:2212.00876: 4 g, R = 5 mm, moment 0.46 A m^2
PAPER = DipolePair.from_sphere(mass=0.004, radius=0.005, moment=0.46)


def test_spin_is_exactly_I_phidot():
    """The dimensionless s_i is I omega_i in units of the angular-momentum scale."""
    rng = np.random.default_rng(3)
    for _ in range(10):
        y = np.concatenate([[rng.uniform(3.0, 8.0)], rng.uniform(-np.pi, np.pi, 3),
                            rng.uniform(-1, 1, 4)])
        st = PAPER.from_dimensionless(y)
        omega1, omega2 = st[10], st[11]
        assert PAPER.inertia * omega1 / PAPER.momentum_scale == pytest.approx(
            y[dyn.IS1], rel=1e-12, abs=1e-15)
        assert PAPER.inertia * omega2 / PAPER.momentum_scale == pytest.approx(
            y[dyn.IS2], rel=1e-12, abs=1e-15)
        # and phi_dot is the angular velocity of the moment's own orientation
        assert PAPER.rhs(0.0, st)[4] == pytest.approx(omega1, rel=1e-12, abs=1e-15)


def test_total_angular_momentum_has_no_intrinsic_term():
    """J = L + I(w1 + w2) exactly -- no gyromagnetic contribution anywhere."""
    y = np.array([4.0, 0.3, 0.8, -1.4, -0.35, 1.1, 0.05, -0.02])
    st = PAPER.from_dimensionless(y)
    orbital = PAPER.mass * ((st[0] * st[7] - st[1] * st[6])
                            + (st[2] * st[9] - st[3] * st[8]))
    assert PAPER.total_angular_momentum(st) == pytest.approx(
        orbital + PAPER.inertia * (st[10] + st[11]), rel=1e-12)


def test_neglected_intrinsic_spin_is_negligible():
    """m/gamma_e is ~2e-8 in problem units, against scan spins of order 0.1-1."""
    assert PAPER.intrinsic_spin == pytest.approx(0.46 / GAMMA_E, rel=1e-12)
    assert PAPER.intrinsic_spin_dimensionless < 1e-6
    # seven orders of magnitude below the smallest spin any scan resolves
    assert PAPER.intrinsic_spin_dimensionless < 1e-4 * 0.1


def test_solid_spheres_touch_at_rho_sqrt5():
    """a = sqrt(2I/m) = sqrt(0.8) R for a solid sphere, so contact is at rho = sqrt(5).

    Parameter-free: independent of size, density and magnetization.  This is where the
    point-dipole treatment stops applying to real spheres.
    """
    for mass, radius, moment in [(0.004, 0.005, 0.46), (7.9, 0.02, 61.0),
                                 (1e-4, 2e-4, 3e-3)]:
        pair = DipolePair.from_sphere(mass, radius, moment)
        assert pair.length_scale == pytest.approx(np.sqrt(0.8) * radius, rel=1e-12)
        assert pair.contact_radius == pytest.approx(np.sqrt(5.0), rel=1e-12)
    assert DipolePair(0.004, 4e-8, 0.46).contact_radius is None
    # I = beta m R^2 generally gives sqrt(2/beta)
    for beta in (2.0 / 5.0, 2.0 / 3.0, 1.0):
        pair = DipolePair.from_sphere(0.004, 0.005, 0.46, beta=beta)
        assert pair.contact_radius == pytest.approx(np.sqrt(2.0 / beta), rel=1e-12)


def test_energy_ceiling_for_a_physically_reachable_barrier():
    """Etil < (3/rho_contact)^3/54, i.e. 0.0447 for solid spheres."""
    assert locked.max_energy_for_barrier_outside(np.sqrt(5.0)) == pytest.approx(
        0.044721, rel=1e-4)
    # at the ceiling the marginal barrier sits exactly at contact
    E = locked.max_energy_for_barrier_outside(np.sqrt(5.0))
    assert locked.barrier_radius(locked.L_max(E)) == pytest.approx(np.sqrt(5.0),
                                                                  rel=1e-9)
    # the absolute bound over all mass distributions (beta <= 1 => rho_c >= sqrt 2)
    assert locked.max_energy_for_barrier_outside(np.sqrt(2.0)) == pytest.approx(
        0.17678, rel=1e-4)


@pytest.mark.slow
def test_contact_and_collapse_criteria_agree_below_the_ceiling():
    """Below the ceiling, reaching contact and collapsing are the same event.

    Above it they are not, so results at larger Etil describe the idealised point-dipole
    problem rather than colliding spheres.
    """
    rc = np.sqrt(5.0)

    def disagreement(E):
        bm = locked.b_max(E)
        b = np.linspace(0.05 * bm, 2.6 * bm, 1500)
        outs = [scatter.run_batch(E, b, 1.9, 1.3, rho_capture=rcap, regularized=True,
                                  rtol=1e-11)["outcome"] for rcap in (1e-2, rc)]
        return float(np.mean(outs[0] != outs[1]))

    assert disagreement(0.03) == 0.0            # comfortably below the ceiling
    assert disagreement(0.3) > 0.05             # well above it
