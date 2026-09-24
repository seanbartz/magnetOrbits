"""Figures for zoomed strip maps (stage 8).

Usage
-----
    python analysis/strip_maps.py results/strip_anti results/strip_mirror

For each strip: the outer collision edge b_out(s, d) and the number of contact intervals as
images (horizontal: offset d across the line; vertical: position s along it), and b_out(d)
profiles at a few s. Written to figures/<name>_edge.pdf, <name>_intervals.pdf, <name>_profiles.pdf.

What is and is not converged (stage 8 findings): b_out agrees with a 4x finer b scan to < 1e-6 on
both strips and the tail-to-tail patch, so those b_out panels are results. Interval counts are only
lower bounds in the escape-window band, where thin windows (and sub-windows) sit below the scan
resolution; the boundary there is fractal (analysis/uncertainty.py). In the fine patch at the band
edge / antiparallel crossing, 1.2% of points also gain a thin contact sliver above b_out under the
finer scan, so b_out is a lower bound there too.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics import central          # noqa: E402
from physics.edges import load_map  # noqa: E402

AXIS_LABEL = {"antiparallel": r"$d/\pi$  ($\phi_2=\phi_1+\pi+d$)", "mirror": r"$d/\pi$  ($\phi_2=\pi-\phi_1+d$)",
              "box": r"$\phi_2/\pi$"}
ALONG_LABEL = {"antiparallel": r"$s/\pi$  ($\phi_1=s$)", "mirror": r"$s/\pi$  ($\phi_1=s$)", "box": r"$\phi_1/\pi$"}


def figures(result_dir, out_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "axes.grid": False, "legend.frameon": False})
    data = load_map(result_dir)
    line, name = data["grid"]["line"], Path(result_dir).name
    along, across = data["along"] / np.pi, data["across"] / np.pi
    # Grid steps from the spec, so a one-row strip (--along-pi LO HI 1) works: along excludes HI.
    a_lo, a_hi, n_along = data["grid"]["along_pi"]
    c_lo, c_hi, n_across = data["grid"]["across_pi"]
    ds = (a_hi - a_lo) / n_along
    dd = (c_hi - c_lo) / (n_across - 1) if n_across > 1 else (c_hi - c_lo) or 1.0
    extent = (across[0] - dd / 2, across[-1] + dd / 2, along[0] - ds / 2, along[-1] + ds / 2)
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parents[1] / "figures"
    out_dir.mkdir(exist_ok=True)
    written = []
    for key, label, cmap in [("outer_edge", r"$b_{\rm out}/R$", "viridis"), ("n_intervals", "contact intervals", "magma")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data[key], origin="lower", extent=extent, aspect="auto", cmap=cmap, interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        if line != "box":                                  # d = 0 is the line itself for strips
            ax.axvline(0, color="w", lw=0.8, ls="--")
        ax.set_xlabel(AXIS_LABEL[line]); ax.set_ylabel(ALONG_LABEL[line])
        fig.tight_layout()
        path = out_dir / f"{name}_{'edge' if key == 'outer_edge' else 'intervals'}.pdf"
        fig.savefig(path); fig.savefig(path.with_suffix(".png"), dpi=110); plt.close(fig)
        written.append(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = np.unique(np.linspace(0, len(along) - 1, 5).astype(int)[1:-1]) if len(along) > 2 else np.arange(len(along))
    for k in rows:
        ax.plot(across, data["outer_edge"][k], "-", lw=1.5, label=fr"$s={along[k]:.3f}\pi$")
    ax.axhline(central.capture_impact_parameter(data["meta"]["settings"]["energy"], 2.0), color="0.4", ls=":", lw=1)
    ax.set_xlabel(AXIS_LABEL[line]); ax.set_ylabel(r"$b_{\rm out}/R$"); ax.legend(fontsize=11)
    fig.tight_layout()
    path = out_dir / f"{name}_profiles.pdf"
    fig.savefig(path); fig.savefig(path.with_suffix(".png"), dpi=110); plt.close(fig)
    written.append(path)
    return written


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        for path in figures(directory):
            print("wrote", path)
