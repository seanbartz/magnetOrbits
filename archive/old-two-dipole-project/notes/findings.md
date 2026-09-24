# Findings so far

Status: plan steps 1–4 complete; the full test suite passes
(`python -m pytest tests -q`).
Everything below is from the dimensionless system; `btil` is in units of
`a = sqrt(2I/m)` and `Etil = E a^3 / C`.

---

## 1. The reduction is exact and has no free parameters  ✔ verified

`notes/derivation.md` §3: rescaling on `a = sqrt(I/mu)` makes the rotational kinetic
coefficient `I/(mu a^2)` identically 1, so the dimensionless Lagrangian

    L = (rho'^2 + rho^2 theta'^2)/2 + (phi1'^2 + phi2'^2)/2 + W(a1,a2)/(2 rho^3)

contains **no parameters at all**.  Only the initial conditions `(Etil, btil, phi1_0,
phi2_0)` remain, exactly as the handoff predicted.

Verified numerically rather than only on paper: `tests/test_dimensionless.py` integrates
three physically different pairs — including one with a deliberately wrong `mu0` and
masses differing by 240x — in full **dimensional lab Cartesian coordinates**, from the
vector force `F1 = (3 mu0/4pi r^4)[...]` and torque `tau = m x B`, and checks they map
onto one dimensionless trajectory (agreement 2e-7 after 60 time units).

The whole orientational dependence is one function,
`W = 2 cos a1 cos a2 - sin a1 sin a2`, with global maximum `W = 2` nose-to-tail.

## 2. Locked-dipole baseline  ✔ reproduced analytically and numerically

    U_eff = L^2/(2 rho^2) - 1/rho^3,  rho* = 3/L^2,  U_eff(rho*) = L^6/54
    btil_max = (27/4)^(1/6) Etil^(-1/3) = 1.374730 Etil^(-1/3)

Numerically, integrating the locked EOM gives capture at `0.999 btil_max` and escape at
`1.001 btil_max` for every `Etil` tested.  The closed-form deflection quadrature agrees
with direct integration to 2e-5.

### Two bookkeeping corrections to the handoff

* **`b_max` dimensional form.**  The coefficient `(27/4)^(1/6)` is right, but it pairs
  with `E^(-1/3)` where `E = mu v^2/2 = m v^2/4` is the *relative-motion* energy.  The
  handoff's `(mv^2)^(-1/3)` is short by `4^(1/3)`; equivalently
  `b_max^6 = 108 C^2/(m^2 v^4)`.  The barrier expressions quoted in the handoff
  (`r* = 3Cm/2L^2`, height `4L^6/27C^2m^3`) are exactly right.
* **Cross-section exponent.**  `sigma ∝ E^(-2/3)` is the *three-dimensional* `pi b_max^2`.
  The planar problem actually being simulated has a cross section with units of length,
  `sigma = 2 b_max ∝ Etil^(-1/3)`.  Worth deciding early which one the paper reports.

### There are no bound orbits

`U_eff` has a barrier maximum and **no minimum** (attractive `1/r^3` is too singular —
L&L §18).  So the only asymptotic outcomes are *escape* and *collapse*; every
"long-lived complex" is a transient.  Recorded as a test.

## 3. `Etil` really is the adiabaticity parameter  ✔

Outermost capture impact parameter, started nose-to-tail (`phi_0 = (0,0)`, i.e. in the
locked basin), relative to the locked prediction:

| `Etil`  | 0.003 | 0.01  | 0.03  | 0.1   | 0.3   | 1.0   |
|---------|-------|-------|-------|-------|-------|-------|
| edge/`btil_max` | 1.031 | 1.064 | 1.111 | 1.107 | 0.983 | 0.847 |

The ratio → 1 as `Etil → 0`, confirming the adiabatic limit.  In between there is a
genuine **~11 % enhancement** peaking around `Etil ≈ 0.03–0.1`, and suppression at large
`Etil` — the qualitative shape the handoff predicted for `sigma(Etil)`.

**But it is strongly orientation-dependent**, which the single-orientation picture hides:

| `Etil` | (0,0) | (0.7,−2.1) | (1.9,1.3) | (π/2,−π/2) |
|--------|-------|------------|-----------|------------|
| 0.03   | 1.111 | 1.030      | 0.937     | 0.926      |
| 0.10   | 1.107 | 0.921      | 0.832     | 0.942      |
| 0.30   | 0.983 | 0.659      | 0.740     | 0.833      |
| 1.00   | 0.847 | 0.420      | 0.618     | 0.674      |
| 3.00   | 0.759 | 0.300      | 0.541     | 0.589      |
| 10.0   | 0.721 | 0.239      | 0.479     | 0.551      |

So the headline `sigma(Etil)/sigma_locked(Etil)` ratio **must** be computed as an average
over `(phi1_0, phi2_0)`, not from a single slice: the spread across orientations at fixed
`Etil` is larger than the effect being measured.

## 4. The capture boundary is fractal — but only in parts of parameter space

Counting disjoint capture intervals in a `chi(btil)` line scan (N = 20 000 per slice)
over a 6 x 4 grid of `(Etil, phi_0)`: **21 of 24 slices give exactly one interval**, with
smooth `chi(btil)` and a smooth time-delay hump (see `figures/step4_deflection.png` (a)).
Three slices are different — `Etil = 0.03 @ (1.9,1.3)`, `Etil = 0.3 @ (1.9,1.3)`,
`Etil = 1.0 @ (0,0)` — and those are exactly the slices with maximum time delays of
4 000–12 000 instead of tens-to-hundreds.

Zooming the `Etil = 0.3, phi_0 = (1.9,1.3)` slice into `btil ∈ [1.4980, 1.5230]` and
refining the grid 64x (`analysis/refine.py`):

| N       | grid db  | intervals | capture measure |
|---------|----------|-----------|-----------------|
| 5 000   | 5.0e-06  | 89        | 3.6607e-03      |
| 20 000  | 1.3e-06  | 179       | 3.8639e-03      |
| 80 000  | 3.1e-07  | 383       | 3.9660e-03      |
| 320 000 | 7.8e-08  | 855       | 4.0138e-03      |

The interval count **does not saturate** — it grows as `N^0.544` — while the total
capture measure converges to ≈ 4.03e-3.  That is a *fat fractal*: a capture set of
positive measure whose boundary has box dimension ≈ 0.54.  `chi(btil)` in this window
shows the textbook chaotic-scattering pattern of singular spikes with interleaved capture
intervals, self-similar under successive zoom (figure panels (c), (d)).

### Control: this is dynamics, not integration noise  ✔

Long-lived chaotic trajectories amplify truncation error, so the same 20 000-point grid
was rescanned under four settings (`analysis/tolerance_control.py`):

| setting                                   | outcomes changed |
|-------------------------------------------|------------------|
| `rtol` 1e-11 → 1e-13                      | 0.035 %          |
| capture radius `rho_c` 1e-2 → 1e-4        | 0.035 %          |
| Sundman regularization off                | 0.035 %          |

7 of 20 000 outcomes move, and those are the points sitting on the boundary itself.  The
insensitivity to `rho_c` also confirms that reaching small `rho` is genuinely terminal.

---

## Caveats on the above

* **"21 of 24 regular" is an upper bound on regularity, not a measurement.**  At N = 20 000
  the grid spacing is ~2.6e-4, while the fractal window found at `Etil = 0.3` is only
  0.02 wide.  A narrower singular window in another slice would simply be missed.  The
  honest statement is "no singular structure *at this resolution*", and the fraction of
  parameter space that is chaotic is currently unmeasured.
* **Interval counting is a crude proxy** for fractality.  The proper measurement is the
  uncertainty exponent `alpha` from `f(eps) ~ eps^alpha` (Ott & Tél), giving the
  uncertainty dimension `D = 1 - alpha`.  The `N^0.544` growth is suggestive, not a
  dimension estimate.
* All scans use **zero initial spin** — see §5 below, which is now quantified.
* `rho0 = 200` is converged for the non-rotating in-state (capture edge moves by
  <0.01 between `rho0 = 200` and `400`; `chi` by 4e-5 between 200 and 800).  It is
  *not* converged once the in-state spins.
* The time-delay distribution (research question 4) has not been measured at all yet.

## 4b. Scope: where the point-dipole model is physically realizable

Each magnet's angular momentum is `s_i = I phi_dot_i` and nothing else — the moment is
rigidly attached, and the electron-spin angular momentum of the magnetization
(`m/gamma_e`, Einstein--de Haas) is excluded by choice.  That term is ~`1.9e-8` in
problem units for the paper's magnets, seven orders below anything resolved here, and is
structurally absent anyway in a hard magnet whose magnetization co-rotates with the
lattice.  `DipolePair.intrinsic_spin` documents it; `tests/test_rigid_rotor_model.py`
pins it.

Because `s = I phi_dot` ties `I` to the mass distribution, the length scale is fixed by
geometry.  For a uniformly magnetized **solid sphere** `a = sqrt(0.8) R`, so two spheres
touch at

    rho_contact = 2R/a = sqrt(5) = 2.2361      (parameter-free)

and requiring the marginal centrifugal barrier `rho* = 3/L_max^2` to lie outside contact
gives `Etil < (3/rho_contact)^3/54`:

| body                        | `beta` | `rho_contact` | `Etil` ceiling |
|-----------------------------|--------|---------------|----------------|
| solid sphere                | 2/5    | 2.236         | **0.0447**     |
| thin shell / surface-loaded | 2/3    | 1.732         | 0.0962         |
| absolute bound (any body)   | 1      | 1.414         | 0.1768         |

Confirmed numerically — capture sets defined by "reach `rho = 1e-2`" vs "reach
`rho = sqrt(5)`" disagree on:

| `Etil`   | 0.0114 | 0.030 | 0.0447 | 0.100 | 0.300 |
|----------|--------|-------|--------|-------|-------|
| outcomes differing | 0.00 % | 0.00 % | 0.10 % | 1.97 % | 17.1 % |

**This re-scopes the results in §4.**  The chaotic slice at `Etil = 0.03` is physically
realizable with solid spheres.  The ones at `Etil = 0.3` and `Etil = 1.0` are *not* — real
spheres would collide first.  They remain valid statements about the idealized
point-dipole problem, which is a legitimate object of study (and `I` is genuinely
independent there), but a paper should say which it is claiming.  The parent paper's own
regime, `Etil = 0.0114`, is comfortably inside the physical window.

## 5. The initial velocity: two parts accounted for, one not

**Translational velocity is fully absorbed.**  The CM velocity is irrelevant (Galilean
invariance) and the relative speed enters *only* through `Etil = v^2/2`; dimensionally
`Etil = v^2 I^(3/2) / (sqrt(2) m^(1/2) C)`, so speed and moment of inertia are the same
knob — tripling `v` at fixed `Etil` means multiplying `I` by `3^(-4/3)`.  `btil = b/a`
carries the transverse part and does not involve `v`.  All verified in
`tests/test_initial_velocity.py` against full dimensional lab-coordinate integrations.

**Initial spin is not.**  `(Etil, btil, phi1_0, phi2_0)` tacitly assumes a *non-rotating*
beam; the complete in-state is six-dimensional,
`(Etil_trans, btil, phi1_0, phi2_0, s1_0, s2_0)`.  `scatter.initial_state` now takes
`s1, s2`, and `initial_state_total_energy` splits a fixed total `Etil` between motion and
spin.

The effect is **large, not perturbative** (`scans/spin_probe.py`, `Etil_trans = 0.3`,
`phi_0 = (1.9, 1.3)`, translational energy held fixed so the baseline is unchanged):

| `(s1,s2)`   | spin KE / E | capture measure | edge/`btil_max` |
|-------------|-------------|-----------------|-----------------|
| (0.0, 0.0)  | 0.000       | 1.277           | 0.740           |
| (0.1, 0.1)  | 0.033       | 0.414           | 0.252           |
| (−0.1,−0.1) | 0.033       | 1.542           | 0.814           |
| (0.1,−0.1)  | 0.033       | 0.780           | 0.578           |
| (0.3, 0.3)  | 0.300       | 1.958           | 1.004           |
| (0.3,−0.3)  | 0.300       | 0.465           | 0.512           |

With **3 % of the energy in spin** the capture measure spans a factor of 3.7 and the
capture edge a factor of 3.2 — and the *sign* of the spin matters as much as its
magnitude (co-rotating vs counter-rotating with the orbital motion).  This is a
leading-order control parameter, comparable to `phi_0` and larger than the `Etil`
enhancement being chased in §3.

### A structural warning that comes with it

With `s_i != 0` there is no asymptotic orientation: the dipole keeps turning on the way
in, so `phi_i_0` becomes a *phase referred to the starting radius `rho0`*.  Measured at
`Etil = 0.3, btil = 3`:

| `rho0`      | 200      | 400      | 800      |
|-------------|----------|----------|----------|
| `chi`, s=0  | 0.082502 | 0.082529 | 0.082543 |
| `chi`, s=0.1| 0.085815 | −0.077531| 0.207638 |

The non-rotating case converges; the spinning one does not, and the capture edge moves
from 0.25 to 0.86 `btil_max` as `rho0` goes 200 → 400.  So a spinning-beam study cannot
hold `phi_0` fixed — the phases must be averaged over (or defined at a fixed reference
radius and reported as such).  Both behaviours are pinned by tests.

## Suggested next steps

1. Uncertainty-exponent measurement on the known-fractal window, then on a grid, to turn
   `N^0.544` into a real `D`.
1b. Decide the scope of the spin axes.  Either declare the non-rotating beam as the
   problem definition (defensible, and it is the case with a clean asymptotic label), or
   commit to phase-averaging over `(phi_0, s_0)` — but the §5 numbers say these cannot
   simply be left at zero without saying so.
2. Map *where* in `(Etil, phi1_0, phi2_0)` the chaotic set lives — a 2-D outcome map at
   fixed `Etil` over `(phi1_0, phi2_0)` is the cheapest view and should show the chaotic
   region directly.
3. Only then `sigma(Etil)`, averaged over orientation, against the `Etil^(-1/3)` baseline.
4. Verify the novelty claim (`notes/literature.md` has an open item on this).
