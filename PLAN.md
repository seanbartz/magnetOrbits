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
- [x] **2. Single trajectory** (`physics/solve.py`, `run_single.py`), $+z/-z$ case.
  *Check:* orientations exactly fixed; $\chi(b)$ vs independent quadrature of the $-1/r^3$
  orbit integral; large-$b$ limit $2/\tilde E\tilde b^3$; drift of $E$, $\mathbf J$, $|\hat\mu|$;
  convergence in rtol and starting radius $r_0$.
- [x] **3. Capture boundary** $\tilde b_c$ by bisection.
  *Check:* both analytic branches at ~6 energies spanning $\tilde E=1/16$; logarithmic
  divergence of $\chi$ as $b\to b_c^+$ (barrier-limited regime).
- [x] **4. Rotational dynamics** (fixed-center mode + free 3D).
  *Check:* head-to-tail small-oscillation frequencies $\omega^2=3/\tilde Ir^3,\,1/\tilde Ir^3$;
  $+z/-z$ scissor-instability growth rate; $E$ and $\mathbf J=\mathbf L+\mathbf S_1+\mathbf S_2$
  conserved in generic 3D runs while $\mathbf L$ alone is not.
- [x] **5. Paper comparison** (Bartz & Shaw, arXiv:2212.00876): moments along the incoming
  $\hat r$, $\tilde E=3\tilde R^3v_i^2\approx0.0335$.
  *Check:* their max colliding $b/R=4.65$ (Fig. 4 caption) vs our value and the locked
  value 5.37. Expected discrepancy (their 4.65 is likely the sticking edge, not the touching
  edge) is reported, not tuned away.
  *Numerics note:* the torque on an incoming non-spinning dipole falls only as $1/r^3$, so an
  orientation specified at a finite $r_0$ differs from the one at infinity by
  $O(1/\tilde I\tilde E r_0)$. Add a first-order incoming-leg correction to the in-state and
  show the $r_0$-convergence order improves from 1 to 2.
- [x] **6b. In-plane orientation maps** at $\tilde E=0.0335$ (approved 2026-09-23).
  Asymptotic angles $(\phi_1,\phi_2)$ from $\hat x$. Per orientation: scan $b\in(0,10]$, bisect
  every sign change → outer edge, capture measure, interval count (`physics/edges.py`,
  `sweep.py`, parallel).
  *Checks:* $+z/-z$ gives one interval $[0,b_c]$; $(0,0)$ gives 5.7546; symmetries
  $(\phi_1,\phi_2)\sim(\phi_1+\pi,\phi_2+\pi)\sim(\phi_2,\phi_1)$ hold on a full-torus coarse map;
  sweep point = direct call; interval counts stable under scan refinement.
- [ ] **6a. $\tilde b_c(\tilde E)$** for head-to-tail vs. locked and $+z/-z$ curves (deferred).

## Findings so far

- **$+z/-z$ is exact but fragile** (stage 4). Scissor growth rate at fixed $r$ is
  $\sigma=1/\sqrt{\tilde I r^3}$ (verified to $10^{-8}$). In scattering, a $10^{-6}$ tilt toward
  $\hat r$ changes the deflection at $\tilde E=0.03$ by $2\times10^{-10}$, $10^{-6}$, $0.04$,
  $4.6$ rad at $b/b_c=2,\,1.05,\,1.01,\,1.001$; at $\tilde E=0.3$ (contact-limited) by
  $\lesssim10^{-10}$. So the power-law $b_c$ is the edge of a measure-zero manifold: for a real
  pair near the barrier-limited edge, the orbiting phase amplifies any tilt enormously.
- Contact-limited branch is $\sqrt{4+\kappa/2\tilde E}$ (an early code version had
  $\sqrt{4+\kappa/\tilde E}$; caught by a test precondition).

- **In-state orientation is now defined at $t\to-\infty$** (stage 5): first-order incoming-leg
  correction raises $r_0$-convergence from order 1 to 2 (generic) and 2 to 3 (head-to-tail);
  residual at $r_0=1000$ is $\sim10^{-7}$ at $\tilde E\approx0.16$ (was $10^{-3}$–$10^{-2}$).
  **Correction (stage 6b):** the residual is $\sim(1/\tilde I\tilde Er_0)^2$ and is amplified where
  the edge depends steeply on orientation. At $\tilde E=0.0335$, $r_0=1000$ leaves edge errors up
  to $5\times10^{-4}$ (relative); orders 1.97–2.02 in $r_0$ confirmed to $6.4\times10^4$. Maps use
  $r_0=64000$ (residual $\le3\times10^{-7}$, +15% cost). `run_trajectory` default unchanged (1000).
- **Paper discrepancy** (stage 5; closed 2026-09-24 at the user's direction, not pursued further; the strict xfail stays as documentation). Head-to-tail at $\tilde E=0.033501$: our collision
  edge $b_c=5.75459$ (converged to $2\times10^{-8}$; single interval; $1.071\times$ the locked
  value 5.3729) vs. the paper's $4.65$. Ruled out: $r_0$, rtol, alignment with $\hat r$ at a
  finite start ($\ge5.66$ even at $8R$), and sticking vs. touching (first-bounce sticking
  estimate with $\Gamma=0.05$: $5.54$). $\tilde E$ larger by $2.07\times$ would give $4.65$
  — suggestive of a factor-2 convention difference; needs the original GlowScript code.
  Checked `Dropbox/ISU/Research/Spherical magnets` (2026-09-23): `magnet_sim.ipynb` is a
  separate 2D sketch (not the paper's code); the draft `equations.tex` defines
  $x_c=(\mu_0\mu^2/4\pi k)^{1/5}=0.59$ mm, without the paper's factor 3 (0.72 mm). If the
  simulation used $\tilde R=R/0.59\,{\rm mm}=8.47$, then $\tilde E=0.0610$ and our touching edge is
  $4.83$ (first-bounce sticking estimate $3.51$), bracketing $4.65$. Unresolved without the code.

- **In-plane orientation map** (6b, $32\times32$ full torus, $\tilde E=0.033501$, $r_0=64000$;
  `results/torus32`, reference `tests/reference/map_torus32.npz`, figures `figures/torus32_*.pdf`).
  Accuracy check (18 points incl. all multi-interval orbits; rtol $10^{-12}$, $2r_0$, $2\times$ scan):
  identical interval structure, edges agree to $4.9\times10^{-7}$. Results:
  outer edge ranges from $4.178$ at $(\phi_1,\phi_2)=(0,\pi)$ (head-to-head, repulsive on approach)
  to $5.756$ near head-to-tail; torus mean $5.565$ (locked-dipole value $5.373$). A low-edge
  "diamond" runs along $\phi_1\pm\phi_2=\pi$ and is narrower than the grid (checkerboard
  alternation along it: unresolved). 992/1024 orientations have one contact interval; 32 (the
  diagonals $\phi_1+\phi_2=15\pi/16$ and its flip image) have an escape window at
  $b\approx2.78$–$3.03$ inside the contact region. No timeouts, no contact at $b_{\max}=10$.

## Cost outlook

Stages 1–5: minutes on one core (scipy). Stage 6: Numba RHS + ~18 workers on the Mac
Studio for $10^5$–$10^6$ trajectories; cluster only near $10^7$.
