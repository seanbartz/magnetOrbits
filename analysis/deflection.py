"""Deflection-function figures: chi(btil), outcome bands, and time delay."""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "scans"))   # line_scan lives in survey.py

from magdip import locked, scatter  # noqa: E402
from survey import line_scan  # noqa: E402


def plot_deflection(ax, b, chi, outcome, b_max_locked=None, s=0.6):
    """chi vs b, with captured impact parameters shown as a band along the bottom."""
    esc = outcome == scatter.ESCAPED
    col = outcome == scatter.COLLAPSED
    ax.scatter(b[esc], chi[esc], s=s, lw=0, color="#1f3b73", rasterized=True)
    if col.any():
        ax.scatter(b[col], np.full(col.sum(), -np.pi * 1.12), s=s, lw=0,
                   color="#b23a2e", rasterized=True)
    if b_max_locked is not None:
        ax.axvline(b_max_locked, color="#888", lw=1.0, ls="--", zorder=0)
    ax.set_ylim(-np.pi * 1.22, np.pi * 1.05)
    ax.set_xlabel(r"$\tilde b$")
    ax.set_ylabel(r"$\chi$")


def figure(path, E, phi1, phi2, n=20_000, spans=None, **kw):
    """chi(btil) with successive zooms on the outermost capture edge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bm = locked.b_max(E)
    r = line_scan(E, phi1, phi2, n=n, **kw)
    spans = spans or _zoom_spans(r)

    fig, axes = plt.subplots(len(spans) + 1, 1, figsize=(8.2, 2.6 * (len(spans) + 1)))
    axes = np.atleast_1d(axes)
    plot_deflection(axes[0], r["b"], r["chi"], r["outcome"], bm)
    axes[0].set_title(r"$\tilde E=%g$, $\phi_1^0=%.2f$, $\phi_2^0=%.2f$   "
                      r"(dashed: locked-dipole $\tilde b_{\max}$)" % (E, phi1, phi2),
                      fontsize=10)
    for ax, (lo, hi) in zip(axes[1:], spans):
        b = np.linspace(lo, hi, n)
        z = scatter.run_batch(E, b, phi1, phi2, regularized=True, rtol=1e-11)
        plot_deflection(ax, b, z["chi"], z["outcome"], bm)
        ax.set_title(r"zoom: $\tilde b\in[%.6f,\ %.6f]$" % (lo, hi), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _zoom_spans(r, levels=2, pad=0.004):
    """Successive windows centred on the outermost capture edge."""
    iv = r["intervals"]
    if not iv:
        return []
    edge = max(hi for _, hi in iv)
    spans = []
    w = pad * edge
    for _ in range(levels):
        spans.append((edge - w, edge + w))
        w /= 40.0
    return spans
