"""Regressions for issues raised in code review of PR #1.

Includes a test for one finding that was *rejected*, so the rejection is backed by
something executable rather than by assertion (see `test_event_not_missed_after_
rejected_steps`).
"""
import os
import shutil

import numpy as np
import pytest

from magdip import dynamics as dyn
from magdip import integrators as itg
from magdip import scatter
from magdip.dynamics import IRHO


# --- scatter API ----------------------------------------------------------------

def test_two_dimensional_sweeps_do_not_crash():
    """`phi1[:, None], phi2[None, :]` broadcasts to (8, n, m); it must be handled.

    This is exactly the orientation map that the findings list as the next step, so it
    has to work.
    """
    p1 = np.linspace(0.0, 1.0, 4)[:, None]
    p2 = np.linspace(-1.0, 1.0, 3)[None, :]
    r = scatter.run_batch(0.3, 2.0, p1, p2, regularized=True, rtol=1e-10)
    assert tuple(r["shape"]) == (4, 3)
    assert r["outcome"].shape == (12,)
    g = scatter.reshape_results(r)
    assert g["outcome"].shape == (4, 3)
    assert g["chi"].shape == (4, 3)
    # the grid must agree with the same points run one at a time
    flat = scatter.run_batch(0.3, 2.0, np.repeat(p1.ravel(), 3),
                             np.tile(p2.ravel(), 4), regularized=True, rtol=1e-10)
    assert np.array_equal(g["outcome"].ravel(), flat["outcome"])


def test_outcome_does_not_depend_on_batch_composition():
    """Per-trajectory timeout horizons: a trajectory must be classified on its own.

    With a batch-wide horizon, a trajectory whose own timeout had elapsed could keep
    running until some *other* member's later horizon and be relabelled.
    """
    E, b, p1, p2 = 0.3, 1.9, 1.9, 1.3
    t_short = 400.0
    alone = scatter.run_batch(E, b, p1, p2, t_max=t_short, regularized=True)
    mixed_t = np.array([t_short, 50_000.0, 50_000.0])
    mixed = scatter.run_batch(E, np.array([b, 4.0, 5.0]), p1, p2, t_max=mixed_t,
                              regularized=True)
    assert mixed["outcome"][0] == alone["outcome"][0]
    assert mixed["t_end"][0] == pytest.approx(float(alone["t_end"][0]), rel=1e-9)


def test_impact_parameter_larger_than_rho0_is_rejected():
    with pytest.raises(ValueError, match="rho0"):
        scatter.initial_state(0.1, 250.0, rho0=200.0)
    with pytest.raises(ValueError, match="positive"):
        scatter.initial_state(0.0, 1.0)
    scatter.initial_state(0.1, 199.0, rho0=200.0)      # allowed


def test_supplied_y0_overrides_stale_E_and_b():
    """A supplied state is authoritative for the delay and timeout bookkeeping."""
    y0 = scatter.initial_state(0.3, 4.0, 0.5, -0.5)
    honest = scatter.run_trajectory(0.3, 4.0, 0.5, -0.5)
    # same state, but nonsense E and b passed alongside it
    lied = scatter.run_trajectory(99.0, 0.01, 0.0, 0.0, y0=y0)
    assert lied["outcome_name"] == honest["outcome_name"]
    assert lied["t_end"] == pytest.approx(honest["t_end"], rel=1e-9)
    assert lied["delay"] == pytest.approx(honest["delay"], rel=1e-6)


def test_asymptotic_parameters_roundtrip():
    for E, b in [(0.1, 3.0), (1.0, 0.5), (0.02, 8.0)]:
        y = scatter.initial_state(E, b, 0.3, -0.7, rho0=5000.0)
        rho, Eo, bo = scatter.asymptotic_parameters(y)
        assert rho == pytest.approx(5000.0)
        assert Eo == pytest.approx(E, rel=1e-7)
        assert bo == pytest.approx(b, rel=1e-7)


# --- integrators ----------------------------------------------------------------

def test_event_not_missed_after_rejected_steps():
    """Rejected steps must not desynchronise the stored event values.

    A review finding claimed the event value recorded inside the event loop leaks into
    the next retry as `g_old`. It does not: `gvals` is resynchronised from the updated
    `(t, y)` after the loop, and rejected columns keep their old state, so they are
    recomputed at that old state. This forces a large initial step so that many steps
    *are* rejected, and checks the results are identical to an unforced run.
    """
    E = 0.1
    bs = np.linspace(0.5, 8.0, 200)
    ref = scatter.run_batch(E, bs, 0.4, -1.1, rtol=1e-12, atol=1e-13)

    y0 = scatter.initial_state(E, bs, 0.4, -1.1)
    evs = [itg.Event(lambda t, y: y[IRHO] - scatter.DEFAULT_RHO_CAPTURE, -1, 2),
           itg.Event(lambda t, y: y[IRHO] - scatter.DEFAULT_RHO0 * (1 + 1e-9), +1, 1)]
    forced = itg.integrate_batch(dyn.rhs, y0, 1e6, evs, rtol=1e-12, atol=1e-13,
                                 h0=500.0)
    assert forced["nrejected"].sum() > 0, "no steps were rejected; test is vacuous"
    assert np.array_equal(forced["outcome"], ref["outcome"])
    esc = ref["escaped"]
    got = scatter.deflection_angle(forced["y"])[esc]
    assert np.max(np.abs(got - ref["chi"][esc])) < 1e-8


def test_unconverged_implicit_step_is_refused():
    """An unconverged stage solve is not a symplectic step and must not be returned."""
    y = np.array([4.0, 0.3, 0.8, -1.4, -0.35, 1.1, 0.05, -0.02])
    with pytest.raises(itg.StageSolveError, match="did not converge"):
        itg.gauss_legendre_step(dyn.rhs, 0.0, y, 1.0, stages=2, max_iter=2)
    loose = itg.gauss_legendre_step(dyn.rhs, 0.0, y, 1.0, stages=2, max_iter=2,
                                    strict=False)
    assert np.all(np.isfinite(loose))
    good = itg.gauss_legendre_step(dyn.rhs, 0.0, y, 1.0, stages=2)
    assert not np.allclose(loose, good)


# --- scan checkpointing ---------------------------------------------------------

@pytest.fixture
def scan_path(tmp_path):
    p = str(tmp_path / "scan.npz")
    yield p
    shutil.rmtree(p + ".parts", ignore_errors=True)


def test_checkpoint_refuses_a_different_sweep(scan_path):
    """Reusing an output path for different inputs must fail loudly, not merge."""
    from runner import CheckpointMismatch, Scan

    b = np.linspace(1.0, 6.0, 40)
    Scan(scan_path, 0.1, b, 0.3, -0.2, chunk=20, regularized=True).run(verbose=False)
    for changed in (dict(E=0.2), dict(phi1=1.0), dict(kw={"regularized": False})):
        kw = changed.pop("kw", {"regularized": True})
        args = dict(E=0.1, b=b, phi1=0.3, phi2=-0.2)
        args.update(changed)
        with pytest.raises(CheckpointMismatch):
            Scan(scan_path, chunk=20, **args, **kw).run(verbose=False)


def test_checkpoint_resumes_and_is_consistent(scan_path):
    from runner import Scan

    b = np.linspace(1.0, 6.0, 40)
    kw = dict(chunk=10, regularized=True)
    first = Scan(scan_path, 0.1, b, 0.3, -0.2, **kw).run(verbose=False)
    assert first["done"].all()
    assert not os.path.exists(scan_path + ".parts")
    again = Scan(scan_path, 0.1, b, 0.3, -0.2, **kw).run(verbose=False)
    assert np.array_equal(first["outcome"], again["outcome"])


def test_partial_scan_resumes_from_shards(scan_path):
    """Killing a run mid-way must leave usable shards, not a corrupt checkpoint."""
    from runner import FIELDS, Scan

    b = np.linspace(1.0, 6.0, 40)
    scan = Scan(scan_path, 0.1, b, 0.3, -0.2, chunk=10, regularized=True)
    # simulate a crash after two chunks: write only those shards
    from runner import run_chunk
    for a, z in [(0, 10), (10, 20)]:
        out = run_chunk(scan.E[a:z], scan.b[a:z], scan.phi1[a:z], scan.phi2[a:z],
                        **scan.kw)
        scan._save_shard(a, z, out)
    assert os.path.isdir(scan_path + ".parts")
    partial = scan._load()
    assert partial["done"][:20].all() and not partial["done"][20:].any()
    full = Scan(scan_path, 0.1, b, 0.3, -0.2, chunk=10,
                regularized=True).run(verbose=False)
    assert full["done"].all()
    assert np.array_equal(full["outcome"][:20], partial["outcome"][:20])
    assert set(FIELDS).issubset(full.keys())
