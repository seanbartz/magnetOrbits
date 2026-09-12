# Dimensionless reduction of two free planar magnetic dipoles

## 1. Setup

Two identical point dipoles, mass `m`, moment of inertia `I` (about the axis
perpendicular to the plane), magnetic moment magnitude `𝔪`.  Motion and moments are
confined to a plane.  Lab-frame orientation angles `φ₁, φ₂` measured from `x̂`.

Work in the centre-of-mass frame with relative coordinate

    r⃗ = r⃗₁ − r⃗₂ = r (cos θ, sin θ),      μ = m/2.

Define the orientation of each moment **relative to the line of centres**

    α_i ≡ φ_i − θ.

## 2. Potential

The dipole–dipole interaction energy

    U = −(μ₀/4πr³) [ 3(𝔪⃗₁·r̂)(𝔪⃗₂·r̂) − 𝔪⃗₁·𝔪⃗₂ ]

with `𝔪⃗_i·r̂ = 𝔪 cos α_i` and `𝔪⃗₁·𝔪⃗₂ = 𝔪² cos(α₁−α₂)` becomes

    3cos α₁ cos α₂ − cos(α₁−α₂) = 2 cos α₁ cos α₂ − sin α₁ sin α₂ ≡ W(α₁, α₂)

so that, with the paper's constant **C ≡ μ₀𝔪²/2π**,

    U = − C W(α₁, α₂) / (2 r³),        W = 2cos α₁ cos α₂ − sin α₁ sin α₂.

`W` is the entire orientational content of the problem.  Its extrema:

| configuration            | (α₁, α₂)       | W   | U            |
|--------------------------|----------------|-----|--------------|
| nose-to-tail (locked)    | (0, 0)         | +2  | −C/r³        |
| nose-to-tail reversed    | (π, π)         | +2  | −C/r³        |
| anti-aligned along r̂     | (0, π)         | −2  | +C/r³        |
| side-by-side antiparallel| (π/2, −π/2)    | +1  | −C/2r³       |

The global minimum `W = 2` is the **locked-dipole** configuration and reproduces the
`U = −C/r³` used in the capture-theory baseline.

## 3. Lagrangian and the unique length scale

    L = ½μ(ṙ² + r²θ̇²) + ½I(φ̇₁² + φ̇₂²) + C W/(2r³)

Two mass-like constants (`μ`, `I`) and one coupling (`C`).  The combination

    a ≡ √(I/μ) = √(2I/m)          (radius of gyration)

is the **only** length that can be built from them, because `C` carries units
energy·length³ and so fixes an energy once `a` is fixed:

    [μ] = M,  [I] = M L²,  [C] = M L⁵ T⁻²
    ⇒  a = √(I/μ),   E₀ = C/a³,   t₀ = √(μ a⁵ / C).

Rescale `ρ = r/a`, `τ = t/t₀`.  Since `I/(μa²) = 1` identically, the rotational kinetic
term becomes `½(φ₁′² + φ₂′²)` with **no coefficient**:

    𝓛̃ = ½(ρ′² + ρ²θ′²) + ½(φ₁′² + φ₂′²) + W(α₁,α₂)/(2ρ³)

**Every material property has dropped out.**  The dimensionless Lagrangian contains no
free parameters at all; the physics is entirely in the initial conditions.

## 4. Hamiltonian and equations of motion

Conjugate momenta `p_ρ = ρ′`, `L ≡ p_θ = ρ²θ′`, `s_i ≡ φ_i′` (spin angular momenta).

    H = p_ρ²/2 + L²/(2ρ²) + (s₁² + s₂²)/2 − W(α₁,α₂)/(2ρ³)

    ρ′  = p_ρ                        p_ρ′ = L²/ρ³ − 3W/(2ρ⁴)
    θ′  = L/ρ²                       L′   = −(W₁ + W₂)/(2ρ³)
    φ_i′ = s_i                       s_i′ = W_i/(2ρ³)

with `W_i ≡ ∂W/∂α_i`:

    W₁ = −2 sin α₁ cos α₂ − cos α₁ sin α₂
    W₂ = −2 cos α₁ sin α₂ − sin α₁ cos α₂

`H` depends on the angles only through `α_i = φ_i − θ` (global rotation invariance), hence

    J = L + s₁ + s₂ = const,        L alone is **not** conserved.

`L′ = −(s₁′ + s₂′)`: orbital angular momentum is exchanged with spin at a rate set by
the non-central part of the force.  This is the crux of the project.

## 5. Control parameters

An asymptotic scattering state (`ρ → ∞`, `s_i = 0`, relative speed `v`, impact
parameter `b`) is fixed by

    Ẽ = E a³/C = v²/2,      b̃ = b/a,      φ₁⁰, φ₂⁰

and nothing else.  Since `a ∝ I^{1/2}`, **Ẽ ∝ E I^{3/2}** at fixed `C`: it is the
adiabaticity parameter.

* `Ẽ → 0`  (small `I`, or slow collision): orientations follow `r̂` adiabatically,
  `W → 2`, the force becomes central `−3C/r⁴` and the problem is integrable.
* `Ẽ → ∞`  (large `I`, or fast collision): orientations frozen, sudden limit, force
  non-central but `W` constant along the encounter.

## 5a. What the rotational degree of freedom is (and is not)

Each magnet has **one** rotational degree of freedom, `φ_i`, and its angular momentum is

    s_i = I φ̇_i

entirely — the angular velocity of the moment's own orientation, with the moment rigidly
attached to the body.  Nothing else contributes.

A real ferromagnet also carries the **electron-spin angular momentum responsible for its
magnetization** (Einstein–de Haas / Barnett), `L_int = 𝔪/γ_e`.  This model excludes it.
Two independent reasons that is safe:

* **Size.**  For the 5 mm NdFeB spheres of arXiv:2212.00876 (`m = 4 g`, `R = 5 mm`,
  `𝔪 = 0.46 A m²`), `L_int = 2.6×10⁻¹² J s`, which is `s ≈ 1.9×10⁻⁸` in the units above —
  seven orders of magnitude below the spins any scan resolves, and equivalent to a body
  rotation of `6.5×10⁻⁵ rad/s`.
* **Structure.**  In a hard magnet the magnetization is pinned to the crystal axes, so it
  co-rotates with the body; there is no rotation of the magnetization *relative to* the
  lattice, which is what the gyromagnetic coupling would act on.

`DipolePair.intrinsic_spin` exists solely to keep this neglected term documented and
checkable rather than silent.

### Consequence: where the point-dipole model stops applying

Because `s_i = I φ̇_i` ties `I` to the body's mass distribution, and `a = √(I/μ)`, the
length scale is fixed by the geometry.  For a uniformly magnetized **solid sphere**,
`I = (2/5)mR²`, so

    a = √(0.8) R = 0.8944 R      ⇒   two spheres touch at   ρ = 2R/a = **√5 ≈ 2.236**

independent of size, density and magnetization.  More generally `I = β m R²` gives
`ρ_contact = √(2/β)`, and `β ≤ 1` for any body, so `ρ_contact ≥ √2` always.

The marginally captured trajectory has its centrifugal barrier at `ρ* = 3/L_max²` with
`L_max = (54Ẽ)^{1/6}`.  Requiring `ρ* > ρ_contact` gives

    Ẽ < (3/ρ_contact)³ / 54

| body                       | `β`   | `ρ_contact` | `Ẽ` ceiling |
|----------------------------|-------|-------------|-------------|
| solid sphere               | 2/5   | √5 = 2.236  | **0.0447**  |
| thin shell / surface-loaded | 2/3   | √3 = 1.732  | 0.0962      |
| absolute bound (any body)  | 1     | √2 = 1.414  | 0.1768      |

Above the ceiling, two extended magnets touch before the barrier is ever reached: the
capture physics is still perfectly well posed for the idealized point-dipole problem, but
is not realizable with colliding spheres.  Verified numerically — the capture sets defined
by "reach `ρ = 10⁻²`" and "reach `ρ = √5`" are *identical* at `Ẽ ≤ 0.03`, differ by 0.1 %
at the `Ẽ = 0.0447` ceiling, and by 17 % at `Ẽ = 0.3`.

The parent paper's regime (`v_i = 1.16 m/s` each, so `Ẽ = 0.0114`) sits comfortably inside
the physical window, with its barrier at `ρ* = 3.53` against contact at `2.236`.

## 5b. What happened to the initial velocity

The asymptotic in-state carries velocity in three places, and they are *not* on the same
footing:

1. **Centre-of-mass velocity** — irrelevant by Galilean invariance; we work in the CM
   frame throughout.
2. **Relative speed `v`** — absorbed entirely into `Ẽ`.  In dimensionless units
   `Ẽ = v²/2` exactly, and dimensionally

       Ẽ = v² I^{3/2} / (√2 · m^{1/2} C)

   so speed and moment of inertia are *the same knob*: tripling `v` at fixed `Ẽ` means
   multiplying `I` by `3^{-4/3}`.  The impact parameter `b̃ = b/a` does not involve `v`
   at all — the transverse part of the velocity is carried by `b̃`, not by a separate
   parameter.  Verified in `tests/test_initial_velocity.py` by integrating two
   dimensionally different experiments (one 3× faster) in full lab coordinates.
3. **Initial spins `s₁⁰, s₂⁰`** — **not** absorbed.  These are two further dimensionless
   parameters, and the `(Ẽ, b̃, φ₁⁰, φ₂⁰)` family of §5 tacitly sets them to zero, i.e.
   assumes a non-rotating beam.  The complete asymptotic in-state is six-dimensional:

       (Ẽ_trans, b̃, φ₁⁰, φ₂⁰, s₁⁰, s₂⁰),     Ẽ_total = Ẽ_trans + (s₁⁰² + s₂⁰²)/2

### Why `s_i⁰ = 0` is a distinguished case, not just a convenient one

With `s_i = 0` the orientation has a genuine limit: the torque falls off as `ρ⁻³`, so
`s_i ~ ρ⁻²` and the accumulated `Δφ ~ ρ⁻¹` converges.  `φ_i⁰` is then a true asymptotic
label and results converge as `ρ₀ → ∞`.

With `s_i ≠ 0` the dipole simply keeps turning on the way in, `φ_i → s_i t`, and there is
no limiting orientation.  `φ_i⁰` degrades into a *phase referred to the particular `ρ₀`
at which the integration was started*, and observables stop converging in `ρ₀`
(measured: `χ` moves by 0.29 between `ρ₀ = 200` and `800`, versus 4×10⁻⁵ at zero spin).
A scan over `(φ₁⁰, φ₂⁰)` at non-zero spin is therefore a scan over arrival phases and
must be treated as something to average over, not a coordinate to hold fixed.

## 6. Locked-dipole (Ẽ → 0) baseline

With `W ≡ 2`, `U = −1/ρ³` and

    U_eff(ρ) = L²/(2ρ²) − 1/ρ³
    ρ*  = 3/L²                              (barrier location)
    U_eff(ρ*) = L⁶/54                        (barrier height)

Capture (fall to the centre) iff `E > L⁶/54`, i.e. `L < L_max = (54 Ẽ)^{1/6}`.
With `L = v b̃ = √(2Ẽ) b̃`,

    **b̃_max = (27/4)^{1/6} Ẽ^{−1/3} ≈ 1.3572 Ẽ^{−1/3}**

Planar ("cross section" = a length):   σ̃ = 2 b̃_max ∝ Ẽ^{−1/3}
3-D analogue (for reference):          σ = π b_max² ∝ E^{−2/3}   (Langevin/locked-dipole)

### Dimensional form and a factor in the handoff

Restoring units, `b_max = (27/4)^{1/6} C^{1/3} E^{−1/3}` where `E = ½μv² = ¼mv²` is the
**relative-motion** energy.  The handoff writes `b_max = (27/4)^{1/6} C^{1/3}(mv²)^{−1/3}`,
which is short by `4^{1/3}`; the correct grouping is `(mv²/4)^{−1/3}`.  Equivalently
`b_max⁶ = 108 C²/(m²v⁴)`.  The barrier expressions quoted in the handoff
(`r* = 3Cm/2L²`, height `4L⁶/27C²m³`) are exactly right and agree with §6 above under
the rescaling — they were used to fix the conventions here.

Also note the handoff's `σ_capture ∝ E^{−2/3}` is the 3-D statement; for the planar
problem actually being simulated the locked-dipole baseline is `σ̃ = 2b̃_max ∝ Ẽ^{−1/3}`.

## 7. Regularization

Near `ρ → 0` with locked dipoles, `½ρ′² ≈ 1/ρ³` ⇒ `ρ ∼ (τ_c − τ)^{2/5}`: collapse in
finite time with `p_ρ ∼ ρ^{−3/2}` diverging.  A Sundman time transformation

    dτ/ds = g(ρ) = ρ^{3/2}

makes `dρ/ds → const`, i.e. removes the blow-up of the leading derivative.  Implemented
as a Poincaré transformation on the extended Hamiltonian

    Γ = g(ρ) (H(q,p) − E₀)

so that the transformed system is still Hamiltonian and a symplectic integrator stays
symplectic in `s`.  `Γ ≡ 0` on the physical manifold; its numerical value is a further
conservation diagnostic.

Unlike a rigid barbell (`I ∝ R²`), here `I` is an independent parameter, so
`ω = s/I` stays finite as `ρ → 0`: the spin degrees of freedom do not blow up, only
the radial ones.
