"""Side-by-side outer-edge maps at the paper energy and at the crossover E = 1/8, one color scale.

Usage
-----
    python analysis/compare_energy_maps.py      # -> figures/torus256_energy_comparison.pdf

Reads the stored 256 x 256 maps (tests/reference/map_torus256.npz, map_torus256_E0.125.npz).
Each panel's locked-dipole edge b_c^locked is marked on the shared color bar.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics import central     # noqa: E402

MAPS = ["map_torus256.npz", "map_torus256_E0.125.npz"]


def main():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 20, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "axes.grid": False})
    maps = [np.load(ROOT / "tests" / "reference" / name) for name in MAPS]
    energies = [json.loads(str(m["settings"]))["energy"] for m in maps]
    vmin = min(np.nanmin(m["outer_edge"]) for m in maps)
    vmax = max(np.nanmax(m["outer_edge"]) for m in maps)

    n = len(maps[0]["phi"])
    step = 2 * np.pi / n
    extent = (-step / 2, 2 * np.pi - step / 2, -step / 2, 2 * np.pi - step / 2)
    ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    labels = ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), layout="constrained")
    for ax, m, energy in zip(axes, maps, energies):
        im = ax.imshow(m["outer_edge"].T, origin="lower", extent=extent, cmap="viridis", aspect="equal",
                       interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(rf"$\tilde E={energy:.4g}$")
        ax.set_xticks(ticks), ax.set_xticklabels(labels)
        ax.set_yticks(ticks), ax.set_yticklabels(labels)
        ax.set_xlabel(r"$\phi_1$")
    axes[0].set_ylabel(r"$\phi_2$")
    cb = fig.colorbar(im, ax=axes, shrink=0.9)
    cb.set_label(r"outer collision edge $b_{\rm out}/R$", labelpad=12)
    for energy in energies:          # locked-dipole value for reference, one per energy
        locked = central.capture_impact_parameter(energy, 2.0)
        cb.ax.axhline(locked, color="w", lw=2)
        cb.ax.text(-0.15, locked, f"locked\n{energy:.4g}", transform=cb.ax.get_yaxis_transform(),
                   ha="right", va="center", fontsize=11)

    out = ROOT / "figures" / "torus256_energy_comparison.pdf"
    fig.savefig(out)
    print(f"color scale {vmin:.3f} .. {vmax:.3f}; wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
