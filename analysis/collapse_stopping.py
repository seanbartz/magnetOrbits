"""Where to terminate an in-spiralling trajectory, and why not deeper.

Reproduces the four measurements behind `notes/collapse_stopping.md`:

1. ``euler_cromer_table``   -- the fixed-step pass-through pathology, and that it does
   *not* improve as the step shrinks (the collapse is a finite-time singularity, so
   any fixed step eventually straddles it).
2. ``energy_vs_stop_radius`` -- ``|dE|/E`` against the stopping radius, against the
   scaling ``rtol / (E rho_stop**3)``.
3. ``barrier_indicator``    -- false-positive/negative rates of the frozen-``(W, L)``
   "past the barrier" test ``kappa = L**2 rho / W < 3/2`` as a collapse classifier.
4. ``step_floor_headroom``  -- how close the adaptive step gets to ``h_min``, below
   which `integrators.integrate_batch` accepts a step regardless of its error.

Run all four with ``python analysis/collapse_stopping.py``; each is a few minutes.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from magdip import dynamics as dyn, integrators as itg, scatter
from magdip.dynamics import IRHO, ITHETA, IPHI1, IPHI2, IPR, IL, IS1, IS2

RHO_CONTACT = np.sqrt(5.0)          # solid spheres; see notes/findings.md 4b


# --- 1. the Euler-Cromer pathology ----------------------------------------------

def euler_cromer(y0, h, nsteps):
    """Semi-implicit (Euler-Cromer) Euler on the dimensionless polar system.

    Momenta are advanced with the force at the current configuration, then the
    coordinates with the *new* momenta.  No stopping rule: this is the scheme the
    parent paper used, run into the singular region on purpose.
    """
    y = np.array(y0, dtype=float)
    rho_min = y[IRHO]
    crossed = False
    for _ in range(nsteps):
        f = dyn.rhs(0.0, y)
        for i in (IPR, IL, IS1, IS2):
            y[i] += h * f[i]
        y[IRHO] += h * y[IPR]
        y[ITHETA] += h * y[IL] / y[IRHO] ** 2
        y[IPHI1] += h * y[IS1]
        y[IPHI2] += h * y[IS2]
        crossed |= y[IRHO] < 0.0
        rho_min = min(rho_min, abs(y[IRHO]))
        if not np.isfinite(y).all():
            break
    return y, rho_min, crossed


def euler_cromer_table(E=0.03, b=1.5, phi=(1.9, 1.3), rho0=50.0, T=300.0,
                       steps=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)):
    y0 = scatter.initial_state(E, b, phi[0], phi[1], rho0=rho0)
    E0 = dyn.energy(y0)
    ref = scatter.run_trajectory(E, b, phi[0], phi[1], rho0=rho0, rho_capture=1e-2)
    print("in-state: E=%g b=%g phi=%s rho0=%g   v_in=%.4f" % (E, b, phi, rho0,
                                                              np.sqrt(2 * E)))
    print("reference (DOP853 + rho<1e-2 event): %s at t=%.4g, dE/E=%.2e\n"
          % (ref["outcome_name"], ref["t_end"], ref["dE"] / abs(E0)))
    print("%-8s %-9s %-11s %-11s %-11s" % ("dt", "rho<0", "rho_min", "E_end/E0", "v_out"))
    for h in steps:
        y, rmin, crossed = euler_cromer(y0, h, int(T / h))
        if np.isfinite(y).all():
            v = float(np.hypot(*dyn.cartesian_relative(y)[2:]))
            print("%-8.0e %-9s %-11.1e %-11.1e %-11.1e"
                  % (h, crossed, rmin, dyn.energy(y) / E0, v))
        else:
            print("%-8.0e %-9s %-11.1e %-11s" % (h, crossed, rmin, "overflow"))


# --- 2. energy error against the stopping radius ---------------------------------

def energy_vs_stop_radius(E=0.1, phi=(0.0, 0.0), n=500, rtol=1e-11,
                          radii=(RHO_CONTACT, 1e-2, 1e-4, 1e-6)):
    """``|dE|/E`` at the stop, against ``rtol / (E rho_stop**3)``.

    The tolerance acts on ``p_rho ~ rho**-3/2`` against a potential ``|U| ~ 1/2rho**3``,
    so the error in the *conserved* energy is amplified by ``|U|/E``.  Irreducible at
    fixed ``rtol``; tightening it buys stopping depth only as the cube root.
    """
    b = np.linspace(0.1, 4.0, n)
    print("%-12s %-8s %-12s %-12s %-12s" % ("rho_stop", "reg", "med|dE|/E", "max|dE|/E",
                                            "rtol/(E r^3)"))
    for rho_c in radii:
        pred = rtol / (E * rho_c ** 3)
        for regd in (False, True):
            r = scatter.run_batch(E, b, phi[0], phi[1], rho_capture=rho_c,
                                  rtol=rtol, regularized=regd)
            d = np.abs(r["dE"] / r["E0"])
            print("%-12.4g %-8s %-12.2e %-12.2e %-12.2e"
                  % (rho_c, regd, np.median(d), d.max(), pred))


# --- 3. the frozen-(W, L) barrier indicator as a classifier ----------------------

def kappa(y):
    """``kappa = L**2 rho / W``; the frozen-``(W, L)`` barrier sits at ``kappa = 3/2``.

    ``U_eff = L**2/2rho**2 - W/2rho**3`` peaks at ``rho* = 3W/(2 L**2)``, so
    ``kappa < 3/2`` with ``p_rho < 0`` means no turning point ahead -- *if* ``W`` and
    ``L`` were constants, which they are not.  ``inf`` where the interaction is
    repulsive and no collapse is possible.
    """
    W = dyn.shape_factor(y[IPHI1] - y[ITHETA], y[IPHI2] - y[ITHETA])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(W > 0.0, y[IL] ** 2 * y[IRHO] / W, np.inf)


def free_fall_indicator(y):
    """``u = -p_rho rho**(3/2) / sqrt(W)``; ``-> 1`` in self-similar free fall."""
    W = dyn.shape_factor(y[IPHI1] - y[ITHETA], y[IPHI2] - y[ITHETA])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(W > 0.0, -y[IPR] * y[IRHO] ** 1.5 / np.sqrt(np.abs(W)), np.nan)


def _run_with_indicator(E, b, phi, rho0=200.0, thresh=1.5):
    y0 = scatter.initial_state(E, b, phi[0], phi[1], rho0=rho0)
    t_max = 200.0 * scatter.free_flight_time(E, b, rho0) + 1000.0

    def ev_cap(t, y):
        return y[IRHO] - 1e-2
    ev_cap.terminal, ev_cap.direction = True, -1

    def ev_esc(t, y):
        return y[IRHO] - rho0 * (1.0 + 1e-9)
    ev_esc.terminal, ev_esc.direction = True, +1

    def ev_kappa(t, y):                       # first inward crossing, non-terminal
        return (kappa(y) - thresh) if y[IPR] < 0.0 else 1.0
    ev_kappa.terminal, ev_kappa.direction = False, -1

    sol = solve_ivp(dyn.rhs, (0.0, t_max), y0, method="DOP853", rtol=1e-11, atol=1e-12,
                    events=(ev_cap, ev_esc, ev_kappa))
    collapsed = bool(sol.t_events[0].size)
    t_end = sol.t_events[0][0] if collapsed else (
        sol.t_events[1][0] if sol.t_events[1].size else sol.t[-1])
    fired = bool(sol.t_events[2].size)
    return collapsed, t_end, (sol.t_events[2][0] if fired else np.nan)


def barrier_indicator(slices=((0.03, (1.9, 1.3)), (0.0114, (0.7, -2.1)),
                              (0.1, (0.0, 0.0))), n=120):
    """False positives/negatives of ``kappa < 3/2`` used as an early-stop classifier."""
    for E, phi in slices:
        bs = np.linspace(0.2, 4.0, n)
        rows = [_run_with_indicator(E, b, phi) for b in bs]
        coll = np.array([r[0] for r in rows])
        t_end = np.array([r[1] for r in rows])
        t_kap = np.array([r[2] for r in rows])
        fired = np.isfinite(t_kap)
        print("\nE=%g phi=%s: %d collapse / %d escape" % (E, phi, coll.sum(),
                                                          (~coll).sum()))
        print("  fired on %d/%d collapses (%d false neg), %d/%d escapes (false pos)"
              % ((fired & coll).sum(), coll.sum(), (~fired & coll).sum(),
                 (fired & ~coll).sum(), (~coll).sum()))
        if (fired & coll).any():
            print("  median time saved %.3g of median t_end %.3g"
                  % (np.median((t_end - t_kap)[fired & coll]), np.median(t_end[coll])))


# --- 4. how close the adaptive step gets to the h_min floor ----------------------

def step_floor_headroom(E=0.1, phi=(0.0, 0.0), n=400,
                        radii=(RHO_CONTACT, 1e-2, 1e-4, 1e-6), h_min=1e-14):
    """Smallest step actually taken, against the floor below which error control is off.

    `integrators.integrate_batch` accepts a step when ``hh <= h_min`` *regardless* of
    its error estimate, so the margin between the two is the margin before an
    uncontrolled step can be taken silently.
    """
    smallest = [np.inf]
    original = itg._dp_stages

    def spy(f, t, y, h):
        pos = np.abs(h[h > 0.0])
        if pos.size:
            smallest[0] = min(smallest[0], float(pos.min()))
        return original(f, t, y, h)

    b = np.linspace(0.1, 4.0, n)
    itg._dp_stages = spy
    try:
        print("%-12s %-8s %-14s %-10s" % ("rho_stop", "reg", "min h used", "h/h_min"))
        for rho_c in radii:
            for regd in (False, True):
                smallest[0] = np.inf
                scatter.run_batch(E, b, phi[0], phi[1], rho_capture=rho_c,
                                  regularized=regd)
                print("%-12.4g %-8s %-14.2e %-10.0f"
                      % (rho_c, regd, smallest[0], smallest[0] / h_min))
    finally:
        itg._dp_stages = original


if __name__ == "__main__":
    print("=" * 72, "\n1. Euler-Cromer without a stopping rule\n", "=" * 72)
    euler_cromer_table()
    print("\n", "=" * 72, "\n2. energy error vs stopping radius\n", "=" * 72)
    energy_vs_stop_radius()
    print("\n", "=" * 72, "\n3. kappa < 3/2 as a collapse classifier\n", "=" * 72)
    barrier_indicator()
    print("\n", "=" * 72, "\n4. adaptive step vs the h_min floor\n", "=" * 72)
    step_floor_headroom()
