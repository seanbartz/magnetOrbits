# PLAN — scattering and capture of two free magnetic spheres

Approved 2026-09-23. Increments are checked off as their validation passes. Each finished
stage also gets a section in `notebooks/progress_summary.ipynb` (explanation, equations,
validation figures from code imported from `physics/`, results table, next steps), committed
with the stage. The notebook covers stages 1–9 as of 2026-09-25.

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
- [x] **6a. Collision edge vs. energy** (approved 2026-09-24; in-plane orientations only).
  16 energies $10^{-3}\le\tilde E\le10$ (log-spaced, incl. the paper value and the crossovers
  $1/16$, $1/8$). Per energy: a $32\times32$ symmetry-reduced map (`analysis/energy_scan.py`), with
  $b_{\max}=\max(10,\,2b_c^{\rm locked})$ at $\Delta b\approx0.05$ and $r_0=\max(1000,\,64000\cdot\tilde E_{\rm paper}/\tilde E)$
  (holds the in-state error $\propto(1/\tilde I\tilde Er_0)^2$ at its validated level).
  Outputs: head-to-tail $b_{\rm out}(\tilde E)$; $\sigma=2\langle b_{\rm out}\rangle$ over orientations, plus
  the capture-measure version; min/max over orientations; fraction with escape windows.
  *Checks:* (1) $+z/-z$ matches the analytic $b_c$ on both branches; (2) head-to-tail
  $b_{\rm out}/b_c^{\rm locked}\to1$ as $\tilde E\to0$ (adiabatic limit); (3) $b_{\rm out}\to2$ as
  $\tilde E\to\infty$, with $(b_{\rm out}^2-4)\tilde E$ finite; (4) at the paper energy the head-to-tail
  edge is 5.7546 and the map matches `map_torus32.npz`; (5) at the extreme energies, 2x $r_0$,
  tighter rtol and a 2x scan on sampled points; orientation-grid error from $16\times16$ subsampling.
  *Results* (all checks PASS; `tests/reference/energy_scan.json`, notebook section 11):
  - $+z/-z$ edge matches the analytic formula to $4.5\times10^{-8}$ at all 16 energies (both branches).
  - Slow limit: head-to-tail $b_{\rm out}/b_c^{\rm locked}=1.008$ at $\tilde E=10^{-3}$, approaching 1 as
    $\tilde E^{0.64}$; the orientation average is $1.0006$ there. The minimum over orientations is
    head-to-head and approaches $(1/2)^{1/3}$ (the $\kappa=1$ law: 0.7964 at $10^{-3}$), as the
    antiparallel invariant family predicts.
  - Head-to-tail beats the locked law by up to $12\%$ at the crossover $\tilde E=1/8$, then falls
    below it (0.96 at $\tilde E=0.5$–1). The orientation average peaks at $+4\%$ near $0.05$ and
    reaches $0.86$ at $\tilde E\approx0.5$.
  - Fast limit: $b_{\rm out}\to2$ ($2.013$ at $\tilde E=10$), with $(b^2-4)\tilde E\to0.51$, i.e.
    $\kappa_{\rm eff}\approx1.03$.
  - Escape windows only for $0.02\lesssim\tilde E\lesssim0.2$ (the grid-limited 3.1% plateau is one
    diagonal of the $32\times32$ grid).
  - Resolution: $2r_0$, rtol $10^{-12}$ and a 2x scan change the maps by $\le7.7\times10^{-7}$ at
    $\tilde E=10^{-3}$, $0.0335$ and 10; orientation-grid error $\le0.8\%$.
- [x] **9. The fractal window band vs. energy** (proposed and run 2026-09-25; in-plane orientations).
  *Question:* how do the extent and the fractal dimension of the escape-window band depend on
  $\tilde E$? At the paper energy the band is $-0.075\pi\lesssim d\lesssim0$ below the mirror line, with a
  sharp low-$d$ edge, a ragged high-$d$ side and $D\approx0.65$–$0.73$ (8c, 8f). The $32\times32$ maps
  (6a) show windows only for $0.02\lesssim\tilde E\lesssim0.2$, but the 3.1% plateau there is one grid
  diagonal, so the grid cannot say how wide the band is, where it opens and closes, or whether
  either end is tied to the locked-law crossovers $\tilde E=1/16$ (barrier to contact) and $1/8$.
  Driver: `analysis/band_scan.py`; references `tests/reference/band_scan.json`, `band_maps.npz`.
  - [x] **9a. Mirror strips across energy.** A one-row probe ($s=\pi/2$, 4x scan) showed that the
    band persists below $\tilde E=0.01$, so the planned range was extended. Strips at 11 energies,
    $\tilde E=0.002$–$0.35$; 32 rows in $s$ ($\pi/32$); in $d$, $\pi/2048$ over $[-\pi/16,\pi/64]$ below
    $\tilde E=0.02$ and $\pi/512$ over $[-3\pi/8,\pi/8]$ above. Each strip was run at the 6a settings and
    with a 4x $b$ scan; band statistics come from the 4x runs.
  - [x] **9b. Uncertainty exponents across energy.** Two band points per row ($s=\pi/4$, $\pi/2$) turned
    out to be too few: one kind of point came out smooth and the other fractal. A **survey** was
    added: 6 window points per row, spread over the band's $d$ range (20000 samples each).
    Controls at every energy: $+z/-z$. The antiparallel line was meant as a second control but is
    not smooth at low energy (see results), so it is reported as a result.
  - [x] **9c. Notebook section 12.**
  *Checks* (all PASS; `analysis/band_scan.py analyze`):
  - (1) at the paper energy, $b_{\rm out}$ matches the 8c strip to $8.2\times10^{-7}$ on 4640 common
    points, with the same sharp edge in every row. The stage-8 points reproduce their counts
    exactly ($\alpha=0.351$, $0.266$).
  - (2) $+z/-z$: $\alpha=1.030\pm0.021$ at all 11 energies. The antiparallel-line structure at
    $\tilde E=0.002$ has Poisson-consistent counts under rtol $10^{-12}$ and $2r_0$.
  - (3) 4x $b$ scan: no row's sharp edge moves. $b_{\rm out}$ moves at 81/81248 points: 80 thin
    contact slivers found above the production edge next to windows, and 1 corner point with
    fractal structure at its edge. A 16x scan confirms the 4x $b_{\rm out}$ at all 81. Points with a
    ~$10^{-5}$-wide undecided (timeout) range, 11 in all, are allowed and listed; their interval
    count is unknown.
  - (4) the band closes between $\tilde E=0.3$ and $0.35$: 9 window points at 0.3 (a pocket near
    head-to-head / tail-to-tail), none at 0.35, also on a $\pi/2048$ grid of 16 rows.
  - (5) strip points on the $32\times32$ grid reproduce the 6a maps (816 points, identical).
  Also: no windows at the ends of the $d$ range; provenance clean, runtime code unchanged.
  *Results:*
  - **Position:** the sharp edge lies below the mirror line by $-d\approx0.52\pi\,\tilde E^{0.54}$ for
    $\tilde E<0.05$, reaching $0.18\pi$ at 0.35. Above $\tilde E\approx0.1$ the band bends into an arc
    in $s$ and breaks up.
  - **Size:** the band's share of in-plane orientations is 0.1% at $\tilde E=0.002$ and peaks at
    **5.2% at $\tilde E=1/8$**. That is the locked law's barrier-to-contact crossover, the same
    energy as the 6a head-to-tail peak. It then falls to 0.85% at 0.25, 0.03% at 0.3, and 0 at 0.35.
  - **Fractality is bimodal:** each surveyed window is smooth ($D\approx0$) or fractal, with
    $D=0.55$–$0.75$ at every energy. The fractal share of survey points is 0/8, 1/12, 2/12,
    4/12, 8/12, 8/12, 9/12, 5/12, 1/4 at $\tilde E=0.002$ … 0.25.
    - Below $1/8$ the fractal part grows from the mirror-line side, and the sharp-edge side stays
      smooth.
    - From $1/8$ up, the side toward the mirror line is smooth and the far side of the arc is
      fractal.
  - **Antiparallel line:** at $\tilde E=0.002$ it has a fat-fractal zone about $2.5\times10^{-3}$ wide
    just below its outer edge ($\alpha=0.40$). The edge count grows 4x per 4x scan refinement,
    while the measure changes by 1.5%. This invariant family is not integrable, so its scattering
    is chaotic near the orbiting edge. From $\tilde E=0.005$ up $\alpha\approx1.04$ on the same
    intervals: the zone, if present, is too thin to register.

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
- **Fine torus map** (2026-10-20, $256\times256$, numba, same settings as the $32\times32$ map;
  `results/torus256_numba`, reference `tests/reference/map_torus256.npz`, figures
  `figures/torus256_*.pdf`). 65536 points in 5.4 min on 18 workers. *Check:* every 8th point is
  the $32\times32$ grid and reproduces `map_torus32.npz` exactly (`test_fine_map_contains_production_map`);
  symmetry partners agree to $2.2\times10^{-6}$ (one orbit of 4 points just above the $2\times10^{-6}$
  tolerance, 30/195584 comparisons above $10^{-6}$; bisection roundoff, no structure mismatches).
  *Result:* the dip and trough are now visible as lines; the minimum is $2.656$ on the antiparallel
  line near tail-to-tail, $(\phi_1,\phi_2)=(0.969\pi,1.969\pi)$ (the stage-8g patches, at finer
  spacing, find $2.13$ nearby). Mean $5.564$, max $5.756$ unchanged; $2456/65536$ ($3.7\%$)
  orientations have an escape window. Interpretation (from the edge lists; $d$ = offset below the
  mirror line in cells of $\pi/128$):
  - Off the two line families the map is a smooth plateau ($45\%$ of orientations within $1\%$ of
    the maximum), so the coarse map's averages were already right.
  - Antiparallel dip: $b_{\rm out}\approx4.72$ on the line, mirror-symmetric across it, rising
    $\propto|d|^{0.52}$ between 1 and 4 cells (stage 8b: $0.57$).
  - Mirror band: the escape window is born at $b=0$ on the mirror line and moves out linearly,
    $b_w\approx0.35$ per cell ($\approx14R$ per radian), widening from $0.01$ at $d=-1$ to $0.37$ at
    $d=-10$, while the outer contact piece above it shrinks ($5.35\to4.45$). The band's sharp low
    edge is where that piece pinches off and $b_{\rm out}$ drops to the window's lower edge
    ($4.45\to3.82$ at $s=\pi/2$).
  - One-sidedness: reflection across the line of approach maps $(\phi_1,\phi_2,b)\to(-\phi_1,-\phi_2,-b)$,
    i.e. $d\to-d$, $b\to-b$. In signed $b$ the window is one line $b\approx-\kappa d$ through $b=0$ on
    the mirror line; $b\ge0$ shows only the $d<0$ half. Hypothesis (untested): it marks a balance
    between orbital angular momentum and the spin from the orientation asymmetry.
  - The minimum is the same pinch-off where the antiparallel dip crosses the band ($d=-8$), not a
    deeper cusp; it sits on a jump, so its value depends on grid spacing (8g).

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
