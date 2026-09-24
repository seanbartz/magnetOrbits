"""Run ONE scattering trajectory and print everything needed to judge it.

This is the permanent inspection tool: any point of any later sweep can be reproduced
here with the same arguments, because sweeps call the same physics.solve.run_trajectory.

Examples
--------
    python run_single.py --energy 0.03 --b 5 --preset antiparallel-z
    python run_single.py --energy 0.03 --b 5 --m1 1 0 0 --m2 1 0 0 --plot
    python run_single.py --energy 0.2 --b 2.9 --preset antiparallel-z --rtol 1e-12

Output (all dimensionless: length R, energy U0 = mu0 mu^2 / 4 pi R^3, time t0):
outcome, closest approach, deflection angle, conservation drifts, and, for configurations
that reduce to a central -kappa/r^3 force, the analytic/quadrature baseline.

Program flow
------------
    run_single.py -> physics.solve.run_trajectory -> scipy solve_ivp
                  -> physics.model (U, F, tau) ; physics.observables (E, J, deflection)
                  -> physics.central (baseline for presets with a central force)
"""

import argparse

import numpy as np

from physics import central
from physics.model import SphereParams
from physics.solve import run_trajectory

# Presets: (m1, m2, kappa of the equivalent central force or None).
PRESETS = {
    # Exact invariant manifold: torques vanish identically, U = -1/r^3.
    "antiparallel-z": ((0, 0, 1), (0, 0, -1), 1.0),
    # Exact too, repulsive U = +1/r^3.
    "parallel-z": ((0, 0, 1), (0, 0, 1), -1.0),
    # Head-to-tail along the incoming direction (the paper's initial condition). NOT a
    # central problem: the moments rotate. The locked limit kappa = 2 is printed for reference.
    "head-to-tail": ((1, 0, 0), (1, 0, 0), 2.0),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--energy", type=float, required=True, help="E_tilde (relative KE / U0)")
    p.add_argument("--b", type=float, required=True, help="impact parameter b / R")
    p.add_argument("--preset", choices=sorted(PRESETS), help="named moment configuration")
    p.add_argument("--m1", type=float, nargs=3, help="moment direction of sphere 1 (lab frame)")
    p.add_argument("--m2", type=float, nargs=3, help="moment direction of sphere 2 (lab frame)")
    p.add_argument("--w1", type=float, nargs=3, default=(0, 0, 0), help="initial spin of sphere 1")
    p.add_argument("--w2", type=float, nargs=3, default=(0, 0, 0), help="initial spin of sphere 2")
    p.add_argument("--beta", type=float, default=0.4, help="I = beta m R^2 (0.4 solid sphere)")
    p.add_argument("--r0", type=float, default=1000.0, help="start (and stop) radius")
    p.add_argument("--rtol", type=float, default=1e-10)
    p.add_argument("--atol", type=float, default=1e-12)
    p.add_argument("--stop", choices=("r_out", "escape"), default="r_out")
    p.add_argument("--plot", action="store_true", help="trajectory and energy exchange plots")
    args = p.parse_args()
    if args.preset is None and (args.m1 is None or args.m2 is None):
        p.error("give --preset or both --m1 and --m2")
    return args


def main():
    args = parse_args()
    kappa = None
    if args.preset:
        m1, m2, kappa = PRESETS[args.preset]
    if args.m1 is not None:
        m1, kappa = args.m1, None
    if args.m2 is not None:
        m2, kappa = args.m2, None

    params = SphereParams(inertia_factor=args.beta)
    res = run_trajectory(args.energy, args.b, m1, m2, w1=args.w1, w2=args.w2, params=params,
                         start_radius=args.r0, stop=args.stop, rtol=args.rtol, atol=args.atol,
                         keep_solution=args.plot)

    print(f"E_tilde = {args.energy:g}   b/R = {args.b:g}   m1 = {np.round(m1, 6)}   m2 = {np.round(m2, 6)}")
    print(f"beta = {args.beta:g} (I_tilde = {params.inertia_tilde:g})   r0 = {args.r0:g}   "
          f"rtol = {args.rtol:g}   atol = {args.atol:g}")
    print("-" * 72)
    print(f"outcome                 {res['outcome']}")
    print(f"closest approach r_min  {res['r_min']:.10f}   (contact at 2)")
    print(f"end time                {res['t_end']:.6g}")
    print(f"deflection (signed)     {res['deflection']:.12f} rad")
    print(f"deflection (3D angle)   {res['deflection_3d']:.12f} rad")
    s_end = res["y_end"]
    print(f"final moments           m1 = {np.round(s_end[6:9], 6)}   m2 = {np.round(s_end[9:12], 6)}")
    print(f"final spins             w1 = {np.round(s_end[12:15], 8)}   w2 = {np.round(s_end[15:18], 8)}")
    print("-" * 72)
    print(f"energy drift            {res['energy_drift']:.2e}  (relative)")
    print(f"J drift                 {res['angular_momentum_drift']:.2e}  (relative)")
    print(f"| |m| - 1 | max         {res['moment_norm_error']:.2e}")
    print(f"RHS evaluations         {res['n_rhs']}")

    if kappa is not None:
        print("-" * 72)
        exact = args.preset in ("antiparallel-z", "parallel-z")
        label = "exact reduction" if exact else "locked-dipole limit only (moments rotate)"
        b_c = central.capture_impact_parameter(args.energy, kappa) if kappa > 0 else float("nan")
        theta = central.deflection_angle(args.energy, args.b, kappa)
        r_turn = central.turning_point(args.energy, args.b, kappa)
        print(f"central -kappa/r^3 baseline, kappa = {kappa:g}  [{label}]")
        print(f"  b_c (collision edge)  {b_c:.10f}")
        print(f"  turning point         {r_turn:.10f}")
        print(f"  deflection (quad)     {theta:.12f} rad   (ours - quad = {res['deflection'] - theta:.2e})")

    if args.plot:
        make_plots(res, params)


def make_plots(res, params):
    import matplotlib.pyplot as plt

    from physics import observables

    plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "axes.grid": False})
    sol = res["solution"]
    # Sample only the interaction region (r < 30) densely.
    t = np.linspace(0, res["t_end"], 20000)
    y = sol.sol(t)
    r = np.linalg.norm(y[0:3], axis=0)
    near = r < 30
    kinetic, rotational, pot = observables.energy_parts(y[:, near], params.inertia_tilde)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(y[0, near], y[1, near], lw=1.5)
    ax1.add_patch(plt.Circle((0, 0), 2.0, fill=False, ls="--"))
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x/R$")
    ax1.set_ylabel(r"$y/R$")
    tt = t[near]
    ax2.plot(tt, kinetic, label="translational KE")
    ax2.plot(tt, rotational, label="rotational KE")
    ax2.plot(tt, pot, label="potential")
    ax2.plot(tt, kinetic + rotational + pot, "k", label="total")
    ax2.set_xlabel(r"$t/t_0$")
    ax2.set_ylabel(r"energy $/U_0$")
    ax2.legend(frameon=False, fontsize=13)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
