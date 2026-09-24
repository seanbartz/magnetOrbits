# PLAN — scattering and capture of two free magnetic spheres

Approved 2026-09-23. Increments are checked off as their validation passes.

## Setup

Two identical, uniformly magnetized spheres (mass $m$, radius $R$, moment $\mu$,
$I=\beta mR^2$, $\beta=2/5$). No contact force and no gravity: a trajectory that reaches
$r=2R$ is a **collision** and is terminated there. Orientations are fully 3D; each moment
$\hat\mu_i$ evolves as $\dot{\hat\mu}_i=\boldsymbol\omega_i\times\hat\mu_i$ (isotropic inertia
of a sphere means only the moment direction matters).

Scales: length $R$, energy $U_0=\mu_0\mu^2/4\pi R^3$, mass $m/2$ (reduced mass),
time $t_0=\sqrt{(m/2)R^2/U_0}$. Dimensionless Hamiltonian

$$H=\tfrac12|\mathbf p|^2+\frac{|\mathbf s_1|^2+|\mathbf s_2|^2}{2\tilde I}
-\frac{1}{r^3}\Big[3(\hat\mu_1\!\cdot\!\hat r)(\hat\mu_2\!\cdot\!\hat r)-\hat\mu_1\!\cdot\!\hat\mu_2\Big],
\qquad \tilde I=2\beta=\tfrac45,\quad r_{\rm contact}=2 .$$

Independent parameters: $\tilde E$ (relative KE $/U_0$), $\tilde b=b/R$, four orientation
angles; initial spins zero. For uniformly magnetized spheres
$\tilde E=\tfrac34\rho v_{\rm rel}^2/\mu_0M^2$ — the sphere size drops out.

Central-force baseline, $U=-\kappa/r^3$ ($\kappa=1$: $+z/-z$; $\kappa=2$: locked head-to-tail):

$$\tilde b_c=\begin{cases}(27\kappa^2/4)^{1/6}\tilde E^{-1/3}, & \tilde E<\kappa/16\\
\sqrt{4+\kappa/2\tilde E}, & \tilde E>\kappa/16\end{cases}
\qquad \chi\to\frac{2\kappa}{\tilde E\tilde b^3}\ (\tilde b\to\infty).$$

Rigorous escape criterion: $\dot r>0$ and $\tfrac12\dot r^2>2/r^3$ (radial dipole force is
bounded below by $-6/r^4$ over all orientations; the centrifugal term only helps).

## Increments

- [x] **0. Scaffold** — `physics/`, `tests/`, README, PLAN, pyproject. *Check:* `pytest` runs.
- [x] **1. Model** (`physics/model.py`): potential, force, torques.
  *Check:* $\mathbf F=-\nabla U$, $\boldsymbol\tau=-\hat\mu\times\partial_{\hat\mu}U$ by finite
  differences at random configurations; $\boldsymbol\tau_1+\boldsymbol\tau_2+\mathbf r\times\mathbf F=0$
  to machine precision.
- [ ] **2. Single trajectory** (`physics/solve.py`, `run_single.py`), $+z/-z$ case.
  *Check:* orientations exactly fixed; $\chi(b)$ vs independent quadrature of the $-1/r^3$
  orbit integral; large-$b$ limit $2/\tilde E\tilde b^3$; drift of $E$, $\mathbf J$, $|\hat\mu|$;
  convergence in rtol and starting radius $r_0$.
- [ ] **3. Capture boundary** $\tilde b_c$ by bisection.
  *Check:* both analytic branches at ~6 energies spanning $\tilde E=1/16$; logarithmic
  divergence of $\chi$ as $b\to b_c^+$ (barrier-limited regime).
- [ ] **4. Rotational dynamics** (fixed-center mode + free 3D).
  *Check:* head-to-tail small-oscillation frequencies $\omega^2=3/\tilde Ir^3,\,1/\tilde Ir^3$;
  $+z/-z$ scissor-instability growth rate; $E$ and $\mathbf J=\mathbf L+\mathbf S_1+\mathbf S_2$
  conserved in generic 3D runs while $\mathbf L$ alone is not.
- [ ] **5. Paper comparison** (Bartz & Shaw, arXiv:2212.00876): moments along the incoming
  $\hat r$, $\tilde E=3\tilde R^3v_i^2\approx0.0335$.
  *Check:* their max colliding $b/R=4.65$ (Fig. 4 caption) vs our value and the locked
  value 5.37. Expected discrepancy (their 4.65 is likely the sticking edge, not the touching
  edge) is reported, not tuned away.
  *Numerics note:* the torque on an incoming non-spinning dipole falls only as $1/r^3$, so an
  orientation specified at a finite $r_0$ differs from the one at infinity by
  $O(1/\tilde I\tilde E r_0)$. Add a first-order incoming-leg correction to the in-state and
  show the $r_0$-convergence order improves from 1 to 2.
- [ ] **6. New phase (re-plan)**: $\tilde b_c(\tilde E)$ and outcome maps over orientations,
  using the symmetries $\hat\mu\to-\hat\mu$ (both) and $1\leftrightarrow2$.

## Cost outlook

Stages 1–5: minutes on one core (scipy). Stage 6: Numba RHS + ~18 workers on the Mac
Studio for $10^5$–$10^6$ trajectories; cluster only near $10^7$.
