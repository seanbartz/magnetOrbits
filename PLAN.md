# PLAN — scattering and capture of two free magnetic spheres

Approved 2026-09-23. Increments are checked off as their validation passes. Each finished
stage also gets a section in `notebooks/progress_summary.ipynb` (explanation, equations,
validation figures from code imported from `physics/`, results table, next steps), committed
with the stage. The notebook covers stages 1–7 as of 2026-09-24.

## Setup

Two identical, uniformly magnetized spheres (mass $m$, radius $R$, moment $\mu$,
$I=\beta mR^2$, $\beta=2/5$). No contact force and no gravity: a trajectory that reaches
$r=2R$ is a **collision** and is terminated there. Orientations are fully 3D; each moment
$\hat\mu_i$ evolves as $\dot{\hat\mu}_i=\boldsymbol\omega_i\times\hat\mu_i$ (isotropic inertia
of a sphere means only the moment direction matters).

Scales: length $R$, energy $U_0=\mu_0\mu^2/4\pi R^3$, mass $m/2$ (reduced mass),
time $t_0=\sqrt{(m/2)R^2/U_0}$. Dimensionless Hamiltonian

$H=\tfrac12\lvert\mathbf p\rvert^2+\frac{\lvert\mathbf s_1\rvert^2+\lvert\mathbf s_2\rvert^2}{2\tilde I}-\frac{1}{r^3}\Big[3(\hat\mu_1\cdot\hat r)(\hat\mu_2\cdot\hat r)-\hat\mu_1\cdot\hat\mu_2\Big]$, with $\tilde I=2\beta=\tfrac45$ and $r_{\rm contact}=2$.

Independent parameters: $\tilde E$ (relative KE $/U_0$), $\tilde b=b/R$, four orientation
angles; initial spins zero. For uniformly magnetized spheres
$\tilde E=\tfrac34\rho v_{\rm rel}^2/\mu_0M^2$ — the sphere size drops out.

Central-force baseline, $U=-\kappa/r^3$ ($\kappa=1$: $+z/-z$; $\kappa=2$: locked head-to-tail):

- $\tilde b_c=(27\kappa^2/4)^{1/6}\tilde E^{-1/3}$ for $\tilde E<\kappa/16$ (barrier-limited)
- $\tilde b_c=\sqrt{4+\kappa/2\tilde E}$ for $\tilde E>\kappa/16$ (contact-limited)
- $\chi\to 2\kappa/(\tilde E\tilde b^3)$ as $\tilde b\to\infty$

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
- [x] **7. Compiled integrator** (`physics/fast.py`, approved 2026-09-24): Numba DOP853 with the
  same tableau, error norm, step control and 7th-order dense output as scipy; events located in
  the loop; in-state correction by fixed-order Gauss–Legendre. scipy path stays as
  `method="reference"`.
  *Checks:* compiled RHS = `equations_of_motion` to roundoff; quadrature = `quad_vec` to
  $10^{-12}$; step sequence and end state match scipy DOP853 (same algorithm) to $\sim10^{-12}$;
  `run_trajectory(method="numba")` reproduces every reference-path test case; recomputed
  $32\times32$ map matches `tests/reference/map_torus32.npz` (identical interval counts, edges
  to $2\times10^{-6}$); measured speedup reported.
- [ ] **8. Zoomed maps of the antiparallel dip and the mirror trough** (started 2026-09-24).
  Strip coordinates (`physics/edges.py`): $s$ along the line, $d$ across it; antiparallel
  $\phi_1=s$, $\phi_2=s+\pi+d$; mirror $\phi_1=s$, $\phi_2=\pi-s+d$. Steps $\pi/64$ along and
  $\pi/512$ ($0.35^\circ$, 32x finer than the torus map) across, chosen so strip points coincide
  with torus points. $\tilde E=0.033501$, $r_0=64000$, numba.
  - [x] **8a. Strip grids** in `sweep.py` (`--strip`, `--along-pi`, `--across-pi`).
    *Check:* strip points on the torus grid reproduce `map_torus32.npz` (incl. a window point);
    resume refuses a changed grid.
  - [x] **8b. Antiparallel strip**, $s\in[0,\pi)$ (64), $d\in[-5\pi/64,5\pi/64]$ (81).
    *Check:* $b_{\rm out}$ identical to $7.7\times10^{-7}$ under a 4x finer $b$ scan (PASS). Interval
    structure differs at 16/5184 points, all near $s\to0,\pi$ where the strip crosses the mirror
    band (see 8c). *Result:* the dip is a smooth cusp, resolved: $b_{\rm out}=4.69$–$4.72$ on the
    line, full width at half depth $0.031\pi$ ($5.6^\circ$), $b_{\rm out}(d)-b_{\rm out}(0)\propto|d|^{0.57}$,
    the same at every $s$.
  - [~] **8c. Mirror strip**, $s\in[0,\pi)$ (64), $d\in[-3\pi/16,3\pi/32]$ (145).
    *Check:* $b_{\rm out}$ identical to $7.7\times10^{-7}$ under the 4x scan (PASS). Interval counts
    and capture measure do **not** converge: the 4x scan changes the structure at 884/9280 points
    (thin escape windows at $b\approx0.15$–$1.35$, width $0.02$–$0.04$, below the $\Delta b=0.05$ scan),
    and at one point a 64x scan splits a window into three (sub-windows at $\Delta b\sim10^{-3}$).
    *Result:* the "trough" is a band $-0.075\pi\lesssim d\lesssim0$ below the mirror line, bounded on
    its low side by a sharp jump in $b_{\rm out}$; escape windows fill the band. Near
    $s\to0,\pi$ (head-to-head / tail-to-tail) $b_{\rm out}$ drops to $2.46$ at
    $(\phi_1,\phi_2)=(0.969\pi,1.975\pi)$, far below the torus minimum 4.18.
  - [ ] **8d. Instability hypothesis:** growth of a small offset from exact antiparallel along the
    trajectory, compared with the measured dip width. *First probe:* the spin-mismatch
    amplification is linear and reaches $A\sim10^2$–$10^3$ near $b_{\rm out}$, which predicts widths
    $\sim10^{-3}$–$10^{-2}$ rad against the measured $0.1$ rad (smooth cusp), so a simple
    $1/A$ picture does not explain the dip.
  - [x] **8f. Fine structure of the window band** (`physics/fractal.py`, `analysis/uncertainty.py`).
    Uncertainty exponent $f(\epsilon)\propto\epsilon^\alpha$ ($D=1-\alpha$), $5\times10^4$ samples per point,
    fit over $\epsilon\le L/30$. *Check:* the controls with one smooth edge give
    $\alpha=0.990\pm0.020$ ($+z/-z$) and $0.971\pm0.017$ (antiparallel line), with counts
    Poisson-consistent with $f=2\epsilon/L$. *Result:* band points give $\alpha=0.351\pm0.003$ and $0.266\pm0.003$ ($D\approx0.65$, $0.73$), so the band has a **fractal boundary**; a two-edge window in the band gives $0.998\pm0.016$. The dimension is local: an earlier run that sampled only the inner 64% of band B's interval (fixed in the PR #7 self-review) gave $0.14$. Edges resolved grow 8 → 214 and 6 → 260 over a 256x scan refinement, while
    the capture measure changes by $\lesssim2$–$3\%$ (fat fractal). Robust to rtol $10^{-12}$ and
    $r_0=128000$ (counts Poisson-consistent; 0.5–0.7% of individual samples flip, as expected
    near a fractal set).
  - [x] **8g. Tail-to-tail corner** (patches at $\pi/512$ and $\pi/8192$ around $(\pi,0)$).
    *Check:* $b_{\rm out}$ converged to $9.3\times10^{-7}$ on the coarse patch. In the fine patch
    (re-centered after PR #7 review; it was off-center) 1.2% of points gain a thin contact sliver
    above $b_{\rm out}$ under the 4x scan, so $b_{\rm out}$ is a lower bound there. *Result:* a low region,
    $b_{\rm out}\approx2.2$–$2.5$, bounded by two sharp edges that meet at the antiparallel cusp; the
    lowest outer edge, $2.13$, lies at that corner, $(\phi_1,\phi_2)\approx(0.975\pi,-0.025\pi)$.
    The deepest pixel went 2.46 (strip) → 2.17 (coarse patch) → 2.13 (fine patch, both runs).
    Exact tail-to-tail is 4.18.
  - [x] **8e. Notebook section 10** for stage 8 and updated summary (section 11), from
    `tests/reference/zoom_maps.npz` and `uncertainty_summary.json`.
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
  to $5.756$ near head-to-tail; torus mean $5.565$ (locked-dipole value $5.373$), so most
  orientations collide beyond the locked value. The low-edge "diamond" is two different features
  (re-analyzed 2026-09-24):
  - a sharp, symmetric dip on the antiparallel lines $\phi_2=\phi_1\pm\pi$ ($b_{\rm out}\approx4.7$
    on the line, $\approx5.35$ one cell away). For $\hat\mu_2=-\hat\mu_1$ the torques are equal, so
    this is an invariant family;
  - a lopsided trough one cell below the mirror line $\phi_1+\phi_2=\pi$ (minimum 4.8–4.9 at
    $15\pi/16$, 5.3 on the line).

  Both are narrower than the grid (unresolved). 992/1024 orientations have one contact interval;
  the 32 in the trough ($\phi_1+\phi_2=15\pi/16$ and its flip image) have an escape window at
  $b\approx2.78$–$3.03$ inside the contact region. No timeouts, no contact at $b_{\max}=10$.

- **Compiled integrator** (stage 7): `method="numba"` takes the same DOP853 steps as scipy
  (identical RHS-evaluation counts on 24 generic runs), end states agree to $\lesssim10^{-12}$,
  in-state quadrature agrees with `quad_vec` to $1.3\times10^{-15}$, and every reference-path
  known result is reproduced. The full $32\times32$ map recomputed with it is identical to the
  scipy map (edges bit-for-bit). Speed: 0.29 ms vs 111 ms per trajectory (383x); the map takes
  6.7 s instead of 34 min on 18 workers.
- **One-transition-per-scan-cell check, whole map** (stage 7 tooling): the $32\times32$ map redone
  with a 4x finer $b$ scan (`n_scan` 800, 25 s with numba) has identical interval counts at all
  1024 points and edges within $6.3\times10^{-7}$ of the production map
  (`analysis/compare_maps.py ... --scan-refinement`): no hidden transitions at 4x resolution.

## Cost outlook

Stages 1–5: minutes on one core (scipy). Stage 6: Numba RHS + ~18 workers on the Mac
Studio for $10^5$–$10^6$ trajectories; cluster only near $10^7$.
