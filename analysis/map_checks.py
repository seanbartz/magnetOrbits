"""Validate and plot an in-plane orientation map produced by sweep.py.

Usage
-----
    python analysis/map_checks.py results/torus32            # checks + figures/<name>_*.pdf

Checks (full-torus runs only, where every point is computed directly)
-------------------------------------------------------------------
Every point is compared with its symmetry partners
    F: (phi1 + pi, phi2 + pi)  global flip,      S: (phi2, phi1)  exchange + flip
(physics/edges.py). Partners must have the same interval count and the same edges to the
bisection tolerance (rel 2e-6, i.e. twice rel_tol), the same edge kinds ("ce"/"ec"),
the same scan-top contact state, and no unresolved (timeout) ranges. CAVEAT: both symmetries hold in floating
point, not only in exact arithmetic (the flip only changes signs; F and tau are term-by-term
symmetric under relabeling m1 <-> m2), so this checks the implementation's symmetry and the
bookkeeping, NOT numerical accuracy. Accuracy is checked by analysis/map_convergence.py.

Figures
-------
outer collision edge b_out(phi1, phi2), capture measure, and number of contact intervals.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics import central                     # noqa: E402
from physics.edges import completeness, load_map, symmetry_orbit   # noqa: E402

EDGE_RTOL = 2e-6


def symmetry_check(data):
    """Max relative edge mismatch over all symmetry partners; count structure mismatches."""
    records = data["records"]
    n = data["meta"]["n_phi"]
    worst, structure_mismatch, compared = 0.0, [], 0
    for (i, j), rec in records.items():
        for a, b in symmetry_orbit(i, j, n):
            if (a, b) == (i, j) or (a, b) not in records:
                continue
            other = records[(a, b)]
            compared += 1
            structure = ("n_intervals", "edge_kinds", "contact_at_scan_top")
            if (any(other[k] != rec[k] for k in structure) or len(other["edges"]) != len(rec["edges"])
                    or other.get("unresolved") or rec.get("unresolved")):
                structure_mismatch.append(((i, j), (a, b)))
                continue
            if rec["edges"]:
                e1, e2 = np.array(rec["edges"]), np.array(other["edges"])
                worst = max(worst, np.max(np.abs(e1 - e2) / e1))
    return worst, structure_mismatch, compared


def make_figures(data, stem):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 20, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "axes.grid": False})
    n = data["meta"]["n_phi"]
    energy = data["meta"]["settings"]["energy"]
    step = 2 * np.pi / n
    extent = (-step / 2, 2 * np.pi - step / 2, -step / 2, 2 * np.pi - step / 2)
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]
    locked = central.locked_capture_impact_parameter(energy)

    panels = [("outer_edge", r"outer collision edge $b_{\rm out}/R$", "viridis", f"{stem}_outer_edge.pdf"),
              ("capture_measure", r"capture measure $/R$", "viridis", f"{stem}_measure.pdf"),
              ("n_intervals", "contact intervals", "magma", f"{stem}_intervals.pdf")]
    written = []
    for key, label, cmap, fname in panels:
        fig, ax = plt.subplots(figsize=(7.2, 6))
        # imshow rows = phi2 (vertical), columns = phi1 (horizontal): "b vs (phi1, phi2)".
        # the outer-edge scale extends to the locked edge (the upper bound observed in stage 6a)
        vmax = max(np.nanmax(data[key]), locked) if key == "outer_edge" else None
        im = ax.imshow(data[key].T, origin="lower", extent=extent, cmap=cmap, aspect="equal",
                       interpolation="nearest", vmax=vmax)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label(label)
        if key == "outer_edge":
            cb.ax.axhline(locked, color="w", lw=2)       # locked head-to-tail edge (spin included), for reference
        ax.set_xticks(ticks), ax.set_xticklabels(labels)
        ax.set_yticks(ticks), ax.set_yticklabels(labels)
        ax.set_xlabel(r"$\phi_1$")
        ax.set_ylabel(r"$\phi_2$")
        fig.tight_layout()
        out = Path(__file__).resolve().parents[1] / "figures" / fname
        out.parent.mkdir(exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        written.append(out)
    return written


def main():
    result_dir = Path(sys.argv[1])
    meta = __import__("json").loads((result_dir / "meta.json").read_text())
    strip = meta.get("grid", {"kind": "torus"})["kind"] == "strip"
    full_torus = not strip and not meta["fundamental"]
    data = load_map(result_dir, expand_symmetry=not full_torus)
    n = data["meta"]["n_phi"]
    energy = data["meta"]["settings"]["energy"]
    shape = "x".join(str(k) for k in data["outer_edge"].shape)
    print(f"{result_dir.name}: {'strip ' + data['grid']['line'] if strip else 'torus'} {shape}, "
          f"E = {energy:.6f}, git {data['meta']['git_hash'][:8]}{' (dirty)' if data['meta']['git_dirty'] else ''}")
    print(f"points: {len(data['records'])} computed, {len(data['errors'])} failed, "
          f"{int(np.isnan(data['outer_edge']).sum())} missing/no-capture")

    # A partial run (--limit, failed points) must not pass or be plotted: the symmetry check only
    # compares partners that exist, so missing points would otherwise go unnoticed.
    # Undecided profiles (timeouts inside capture_profile) are successful records with
    # unresolved ranges; they must not pass either, on the torus or on a strip.
    c = completeness(data)
    if any(c.values()):
        print(f"INCOMPLETE map: {len(c['missing'])} expected points missing (e.g. {c['missing'][:5]}), "
              f"{len(c['failed'])} failed, {len(c['unresolved'])} with unresolved (timeout) ranges "
              f"(e.g. {c['unresolved'][:5]})  -> FAIL")
        raise SystemExit(1)
    if strip:
        counts = {int(k): int((data["n_intervals"] == k).sum()) for k in np.unique(data["n_intervals"])}
        print(f"complete; outer edge {np.nanmin(data['outer_edge']):.4f} .. {np.nanmax(data['outer_edge']):.4f}; "
              f"interval counts {counts}; timeouts {int(data['n_timeouts'].sum())}. "
              "Strip figures: analysis/strip_maps.py")
        return

    if full_torus:
        worst, mismatches, compared = symmetry_check(data)
        status = "PASS" if worst < EDGE_RTOL and not mismatches else "FAIL"
        print(f"symmetry check: {compared} partner comparisons, max rel edge diff {worst:.1e} "
              f"(tol {EDGE_RTOL:.0e}), structure mismatches {len(mismatches)}  -> {status}")
        for pair in mismatches[:10]:
            print("   mismatch", pair)
        if status == "FAIL":
            make_figures(data, result_dir.name)
            raise SystemExit(1)

    outer = data["outer_edge"]
    nint = data["n_intervals"]
    locked = central.locked_capture_impact_parameter(energy)
    antiparallel_z = central.capture_impact_parameter(energy, 1.0)
    i_max, j_max = np.unravel_index(np.nanargmax(outer), outer.shape)
    i_min, j_min = np.unravel_index(np.nanargmin(outer), outer.shape)
    print(f"outer edge: min {np.nanmin(outer):.4f} at phi = ({data['phi'][i_min]:.3f}, {data['phi'][j_min]:.3f}), "
          f"max {np.nanmax(outer):.4f} at ({data['phi'][i_max]:.3f}, {data['phi'][j_max]:.3f}), "
          f"torus mean {np.nanmean(outer):.4f}")
    print(f"  reference: locked head-to-tail (spin included) {locked:.4f}; +z/-z (kappa=1) {antiparallel_z:.4f}")
    counts = {k: int((nint == k).sum()) for k in np.unique(nint)}
    print(f"contact-interval counts over the grid: {counts}")
    print(f"timeouts: {int(data['n_timeouts'].sum())}; contact at scan top: "
          f"{sum(r['contact_at_scan_top'] for r in data['records'].values())}")
    for path in make_figures(data, result_dir.name):
        print("wrote", path)


if __name__ == "__main__":
    main()
