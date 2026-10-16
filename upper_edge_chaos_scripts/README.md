# Upper-edge chaos study (exploratory scripts)

Exploratory scripts for the escape window in the mirror band at
s = π/4 (φ₁ = π/4, φ₂ = 3π/4 + d), Ẽ = 0.0335. They compare a regular
window edge (cut B, d = −0.070π: a periapsis staircase onto one unstable
orbiting state) with a fractal edge (cut A, d = −0.042π: windows nested in
windows, uncertainty exponent α < 1).

These are working scripts, not a polished pipeline. They are kept for
provenance of the figures.

## Assumptions

- Run from the repo root so that `physics` imports. Each script does
  `sys.path.insert(0, 'scratch')` and reads/writes `scratch/*.npz|json|pkl`,
  so copy `probe.py` into `scratch/` first (or put this directory on the path
  and create `scratch/` for the outputs).
- `figs.py` writes to `OUT='/mnt/user-data/outputs/'`; change it before running locally.
- The pools use 2 workers (`Pool(2)`).

## Pieces

| Script | Role |
|---|---|
| `probe.py` | Shared helper: outcome (0 escape, 1 contact, 2 timeout) and periapsis count for one trajectory using `physics.fast.integrate_scattering`; `mirror(s, d)` builds the moment pair |
| `timing.py`, `xcheck.py`, `map_conv.py`, `chk.py` | Sanity checks: speed, start radius, outcome edges vs. `run_trajectory` |
| `bd_image.py` → `scratch/bd_s025.npz` | (b, d) outcome map over the band |
| `bd_zoom.py`, `look.py` | Zoomed maps and quick-look plots |
| `upper_survey.py` | Counts transitions near the upper window edge as a function of d |
| `cut.py`, `lookcut.py` | 1-D cuts in b at fixed d (`cA0_12`, `cA1_12`, `cA2_12`, `cB0_12`, ...) |
| `stair.py`, `stair2.py` | Bisect the cut-B edge and sample the periapsis staircase below it (`stairB*.npz`) |
| `rtolchk.py` | Tolerance checks for cut-A outcomes and the cut-B edge location |
| `po.py`, `poA.py` | Per-periapsis state along trapped orbits near the B and A edges |
| `unc.py`, `unc_d.py` | Uncertainty exponent for cuts A/B and across d (`unc.json`, `unc_d.json`) |
| `trajs.py`, `echk.py` | Dense trajectories for the figures (`trajs.pkl`) and the energy split at apoapsis |
| `figs.py` | Figures 1–3: band overview, regular vs. fractal edge, trapped orbits |

Rough order: `bd_image` → `upper_survey` / `cut` → `stair2 1e-13` → `unc`, `unc_d` → `trajs` → `figs`.
