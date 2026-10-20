# APS EGLS 2026 talk: figures by slide

Each file is named `Slide N - <name>`, where `<name>` is the file the generating script writes.
Missing figures are marked NEEDED.

Where the figures come from:

- The `talk_*` scripts in `upper_edge_chaos_scripts/` generate the slide figures; they write to
  `upper_edge_chaos_scripts/output/` (or `UEC_OUT`) and read the pipeline outputs in that folder's
  `data/` (see its README):
  - `talk_effpot.py`: `effective_potential.pdf`, `orientation_trajectories.pdf`
  - `talk_figs.py`: `band_overview_bd.pdf`, `window_trajectories.pdf`, `staircase.pdf`,
    `rocking_orbit.pdf`, `rocking_orbit_trajectory.pdf`, `fractal_zooms.pdf`, `neighbor_pair.pdf`,
    `uncertainty_vs_d.pdf`, `spin_reservoir.pdf`
  - `talk_ueff_frozen.py`: `frozen_ueff_head_to_tail.pdf`, `frozen_ueff_window.pdf`,
    `frozen_ueff_rocking.pdf`, `frozen_ueff_bound_excursion.pdf`
  - `talk_ueff_gifs.py`: `frozen_ueff_head_to_tail.gif`, `frozen_ueff_window_b3.0.gif`,
    `frozen_ueff_window_b3.2.gif`, `frozen_ueff_window_b3.4.gif`, `frozen_ueff_rocking.gif`,
    `frozen_ueff_bound_excursion.gif`
- The rest (`energy_scan_edges`, `torus32_*`, `strip_mirror_scan800_*`, `patch_min_scan800_intervals`,
  `uncertainty_exponent`, `uncertainty_resolution`, `band_scan_*`) are copies of files in `../figures`
  (originals untouched).

The "frozen U_eff" figures show the instantaneous effective potential at frozen orbital angular
momentum and moment orientations, U_eff(r; t) = L(t)^2 / 2r^2 - C(t) / r^3, with the system drawn
as a point at (r, E - K_rot). The GIFs animate the same picture with a space view and the path in
(r, E - K_rot).

| Slide | Topic | File(s) | What it shows |
|---|---|---|---|
| 1 | Title | none | |
| 2 | Hook: stick or fly apart? | NEEDED | Photo or still of two magnets rolling past each other (Lindén et al. 2018 setup) |
| 3 | Model and parameters | NEEDED | Schematic: two spheres, b, moment angles phi1 and phi2, contact at r = 2R |
| 4 | Locked law with spin vs free rotors | energy_scan_edges.pdf | Left: b_out vs E with locked law and contact branch. Right: b_out / b_locked for head-to-tail and the orientation average |
| 4 | | effective_potential.pdf | Effective potentials: locked head-to-tail with co-rotating spin (the adiabatic limit) and moments perpendicular to the collision plane |
| 4 | | orientation_trajectories.pdf | Trajectories at one (E, b) for locked head-to-tail and out-of-plane antiparallel moments |
| 4 | | frozen_ueff_head_to_tail.pdf, frozen_ueff_head_to_tail.gif | Frozen U_eff for a free-rotor head-to-tail capture at b = 5R (compare the static locked curve) |
| 5 | Capture edge vs orientation | torus32_outer_edge.pdf, torus32_cuts_dphi.pdf | Full (phi1, phi2) map of b_out; cuts showing the antiparallel dip and the trough below the mirror line |
| 6 | Escape windows | torus32_intervals.pdf, strip_mirror_scan800_intervals.pdf, strip_mirror_scan800_profiles.pdf, band_overview_bd.pdf, window_trajectories.pdf | Where windows occur on the torus; zoom on the mirror band; the window tongue in (b, d); trajectories below / inside / above the window |
| 6 | | frozen_ueff_window.pdf, frozen_ueff_window_b3.0.gif, frozen_ueff_window_b3.2.gif, frozen_ueff_window_b3.4.gif | Frozen U_eff at d = -0.07pi below (b = 3.0R, contact), inside (3.2R, escape) and above (3.4R, contact) the window |
| 7 | Regular edge | staircase.pdf, rocking_orbit.pdf | Periapsis staircase vs b_edge - b; r(t) and moment angles of the rocking orbit |
| 7 | | rocking_orbit_trajectory.pdf | Five snapshots over one rocking period of the orbit in space, with the moments |
| 7 | | frozen_ueff_rocking.pdf, frozen_ueff_rocking.gif | Frozen U_eff for the rocking orbit, 3e-9 below the cut-B edge |
| 8 | Fractal edge | uncertainty_exponent.pdf, patch_min_scan800_intervals.pdf, fractal_zooms.pdf, uncertainty_vs_d.pdf, neighbor_pair.pdf | Controls vs bands; regular cut vs fractal cut with zooms x16 and x400; f(eps) across d (onset near -0.052pi); r(t) of two b values 2.5e-8 apart |
| 9 | Energy dependence | band_scan_vs_energy.pdf | Band position, size, uncertainty dimension and fractal share vs E |
| 10 | Spin as an energy reservoir | spin_reservoir.pdf | r(t) and K_rot/E for a trapped run (K_rot = 1.04 E at an 11R apoapsis) |
| 10 | | frozen_ueff_bound_excursion.pdf, frozen_ueff_bound_excursion.gif | Frozen U_eff for a bound excursion on the fractal side (the contact member of the neighbor pair) |
| 11 | Summary | none | |
| B1 | Bartz & Shaw factor-2 check | NEEDED | fig4_coupling_check.png (from the 2026-10-15 session; not in the repo) |
| B2 | Convergence | uncertainty_resolution.pdf, band_scan_strips.pdf | Edges resolved and capture measure vs scan spacing; strips at all 11 energies |
| B3 | Analytic benchmark | energy_scan_edges.pdf | +z/-z integrator vs exact locked law (agree to 4.5e-8) |

## Force figures/

Dipole-dipole geometry and force figures (not yet assigned to a slide; candidates for slide 3).
`make_dipole_figs.py` (in that folder) writes the TikZ sources `fig*.tex`; the PNGs are rendered
from them.

| File(s) | What it shows |
|---|---|
| fig1_geometry.tex, fig1_geometry.png | Two spheres with moments mu_1 and mu_2 and the separation direction r_hat |
| fig2_field_energy.tex, fig2_field_energy.png | Field lines of dipole 1, its field B_1 at dipole 2, and the interaction energy |
| fig3_force_terms.tex, fig3_force_terms.png | The force on dipole 2 split into its central and two moment-direction terms |
| fig4_special_cases.tex, fig4_special_cases.png | Force and energy for head-to-tail, side by side (parallel and antiparallel) and T configurations |
| make_dipole_figs.py | Generates the four TikZ sources |
