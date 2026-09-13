# Literature notes

## Directly relevant

**Bartz & Shaw, arXiv:2212.00876** — *Numerical simulation of non-central collisions of
spherical magnets.*  The dissipative parent of this project: same two-dipole system, but
with a damped-spring contact force and Euler–Cromer integration, tied to a physical
demonstration.  The present repo strips the contact force and the damping, which is what
makes the problem universal (see `notes/derivation.md`: all material parameters drop out).
PDF and LaTeX source are under `paper/`.

### Possible erratum in the parent paper

A code review of PR #1 flagged the sentence following the kinetic-energy ratio
(`arxivVersion.tex` line 209), and the algebra appears to support it.  The paper gives

    K_f/K_i = (1/2beta) (b/R)^2

and then states "the final kinetic energy exceeds the initial when `b/R > 1/sqrt(2 beta)`".
But `K_f/K_i > 1` requires `(b/R)^2 > 2 beta`, i.e.

    b/R > sqrt(2 beta)          [not 1/sqrt(2 beta)]

With the barbell value `beta = 14/5` the threshold is `sqrt(5.6) = 2.366`, not
`1/sqrt(5.6) = 0.423` — a difference that matters, since the reported simulations run out
to `b/R = 4.65` and therefore straddle the correct threshold rather than lying entirely
above the incorrect one.  The displayed equation itself is fine; only the inequality that
interprets it is inverted.

**The archival source under `paper/` is deliberately left unmodified** — it is a verbatim
copy of the arXiv submission, kept for reference.  Worth checking independently and, if
confirmed, folding into any revision.

(The same review also reported that `\Dot`/`\Ddot` are undefined and prevent the paper
from compiling.  That is incorrect: they are amsmath's `\mathpalette` accent forms, and
`pdflatex arxivVersion.tex` completes cleanly from `paper/source/`.)

**Edwards, Riffe, Ji & Booth, Am. J. Phys. 85, 130 (2017)** — establishes that a uniformly
magnetized sphere is *exactly* a point dipole for external fields and forces.  This is what
licenses the point-dipole treatment here, and also why `I` is an independent parameter
rather than being tied to the interaction: for a sphere `I = (2/5)MR^2`, but nothing in the
interaction refers to `R`.

**Haugen & Edwards, Am. J. Phys. 88, 365 (2020)** — two freely rotating dipoles with
**fixed centres**.  Orientation dynamics only; `r` is a parameter, not a coordinate.

**Schmelcher group, arXiv:1609.05677** — two rigid rotors, again **fixed centres**, chaotic
energy transfer between the rotational degrees of freedom.

**arXiv:nucl-th/9212009** — chaotic scattering of deformed nuclei.  The closest structural
analogue: the Hamiltonian skeleton `H = p^2/2m + I^2/2J + (L-I)^2/2mr^2 + V(r,theta)` has
the same orbital/spin exchange, and the same consequence that only the total angular
momentum is conserved.

**Ott & Tél, Chaos 3, 417 (1993)** — methodology for chaotic scattering: uncertainty
dimension, singular sets of the deflection function, algebraic vs exponential decay of
time-delay distributions.

**Landau & Lifshitz, *Mechanics* §18** — "fall to the centre".  For `U ~ -alpha/r^n` with
`n > 2` the particle reaches the origin in finite time; this is why the `rho -> 0` collapse
here is a genuine terminal outcome rather than a numerical artefact, and why `U_eff` has a
barrier maximum and *no* minimum (hence no bound orbits — recorded as a test in
`tests/test_conservation.py`).

## Capture-theory baseline

The `W -> 2` limit is the classical **locked-dipole** capture model of ion–molecule
reaction dynamics (the permanent-dipole analogue of Langevin capture).  Standard results:
`U_eff = L^2/2mu r^2 - C/r^3`, barrier at `r* = 3C/(2 L^2/mu)`, and a 3-D capture cross
section `sigma = pi b_max^2 ∝ E^(-2/3)`.  The planar problem simulated here has
`sigma = 2 b_max ∝ E^(-1/3)`.

## Open item: novelty check

The handoff's claim — that the **free–free** case (both centres *and* both orientations
free) is open, while the closest prior work fixes the centres — is consistent with the
references above, but has **not** been independently verified by a literature search in
this repo.  Do that before any write-up.  Specific things to search for:

* "dipole–dipole scattering" in the atomic/molecular collision literature, where two free
  polar molecules colliding is a standard problem (Bohn, Ticknor, Julienne on
  dipolar collisions) — much of that is quantum, and usually with the dipoles either
  fixed in space by an external field or treated as rigid rotors in a full close-coupling
  treatment, but the *classical* free-rotor version may exist.
* Granular / magnetic-colloid literature, where clusters of free magnetic spheres are
  simulated routinely, though normally with contact forces and dissipation.
* Chaotic-scattering literature for "rotational–translational" or "internal degree of
  freedom" scattering beyond the nuclear case.
