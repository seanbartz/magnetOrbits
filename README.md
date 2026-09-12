# magnetOrbits — conservative scattering of two free magnetic dipoles

Classical scattering of two **free** rigid magnetic dipoles in a plane: both translating
and rotating, no dissipation, no contact force.  This is the conservative, universal
version of the collision studied in [Bartz & Shaw, arXiv:2212.00876](https://arxiv.org/abs/2212.00876),
with the damped-spring contact force stripped away.

Closest prior work (Haugen & Edwards 2020; Schmelcher group 2016) keeps the dipole
**centres fixed**.  Here both centres and both orientations are free, so orbital angular
momentum `L` is *not* conserved — only `J = L + s1 + s2` is.  That is the crux: a
trajectory can dump orbital angular momentum into spin and pass *under* the centrifugal
barrier that would otherwise reflect it.

## The result that makes the project tractable

Non-dimensionalizing on the radius of gyration `a = sqrt(2I/m)` and the coupling
`C = mu0 m^2 / 2pi` removes **every** material property.  The dimensionless Hamiltonian

    H = p_rho^2/2 + L^2/(2 rho^2) + (s1^2 + s2^2)/2 - W(a1, a2)/(2 rho^3)
    W = 2 cos a1 cos a2 - sin a1 sin a2,      a_i = phi_i - theta

has **no free parameters at all**.  The entire problem is the four-dimensional family of
initial conditions `(Etil, btil, phi1_0, phi2_0)`, and `Etil ∝ E I^(3/2)` is the
adiabaticity parameter.  See [notes/derivation.md](notes/derivation.md).

This is verified numerically, not just on paper: `tests/test_dimensionless.py` integrates
three physically different dipole pairs (different masses, inertias, moments, and even a
different `mu0`) in full dimensional lab Cartesian coordinates and checks that they map
onto one dimensionless trajectory.

## Layout

    src/magdip/
        dynamics.py     dimensionless EOM, energy, J, the shape factor W
        dimensional.py  independent Cartesian reference built from F = ..., tau = m x B
        locked.py       Etil -> 0 baseline: barrier, b_max, deflection by quadrature
        regularize.py   Sundman/Poincare time transform for the rho -> 0 collapse
        integrators.py  vectorised adaptive Dormand-Prince 5(4); symplectic Gauss-Legendre
        scatter.py      initial conditions, events, observables, batch driver
    tests/              conservation, dimensionless reduction, locked limit, integrators
    scans/              checkpointed, resumable sweeps
    analysis/           deflection functions, cross sections, dimension estimates
    notes/              derivations and literature notes

## Usage

```bash
python -m pytest tests -q              # 54 tests, ~13 s
python -m pytest tests -q -m slow      # the physics-validation subset
```

```python
from magdip import scatter, locked
import numpy as np

b = np.linspace(0.1, 8.0, 20_000)
r = scatter.run_batch(Etil=0.1, b=b, phi1=0.7, phi2=-2.1, regularized=True)
r["escaped"], r["chi"], r["delay"], r["spin_fraction"]

locked.b_max(0.1)          # 2.9618 -- the central-force baseline
```

A checkpointed sweep, resumable after a kill, parallel over chunks:

```bash
python scans/deflection_scan.py --E 0.1 --n 40000 --workers 20 --out data/chi.npz
```

Throughput is ~2 ms per trajectory per core, so 10^7 trajectories is ~35 min on 20 cores.

## Baseline

In the locked-dipole (`Etil -> 0`) limit the moments pin nose-to-tail, `W -> 2`, the force
becomes central `-3/rho^4`, and capture is a sharp threshold:

    U_eff = L^2/(2 rho^2) - 1/rho^3,   rho* = 3/L^2,   U_eff(rho*) = L^6/54
    btil_max = (27/4)^(1/6) Etil^(-1/3)

Dimensionally `b_max = (27/4)^(1/6) C^(1/3) E^(-1/3)` with `E = mu v^2/2 = m v^2/4` the
**relative-motion** energy.  The planar capture cross section is a length,
`sigma = 2 b_max ∝ Etil^(-1/3)`; the familiar `E^(-2/3)` is the three-dimensional
`pi b_max^2`.

## Status

Plan steps 1–4 are complete and under test.  The headline so far: the capture boundary is
**fractal in parts of parameter space but not generically** — 21 of 24 surveyed
`(Etil, phi_0)` slices are regular at 20 000-point resolution, while the others show a fat
fractal whose interval count grows as `N^0.544` under a 64x grid refinement, robust to a
100x tolerance change.  See [notes/findings.md](notes/findings.md) and
`figures/step4_deflection.png`.
