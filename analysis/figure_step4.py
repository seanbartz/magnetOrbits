"""Step-4 figure: regular vs singular deflection functions.

Top row: a representative *regular* slice -- one capture interval, smooth chi(btil).
Lower rows: a *singular* slice and two successive zooms on the fine structure beyond
the main capture interval.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from magdip import locked, scatter  # noqa: E402


def _chunk(args):
    E, b, p1, p2, kw = args
    r = scatter.run_batch(E, b, p1, p2, **kw)
    return np.asarray(r["outcome"]), np.asarray(r["chi"]), np.asarray(r["delay"])


def scan(E, p1, p2, lo, hi, n, workers=20, **kw):
    kw.setdefault("regularized", True)
    kw.setdefault("rtol", 1e-11)
    b = np.linspace(lo, hi, n)
    if workers <= 1 or n < 4000:
        o, c, d = _chunk((E, b, p1, p2, kw))
    else:
        import multiprocessing as mp
        parts = np.array_split(b, workers * 2)
        with mp.Pool(workers) as pool:
            res = pool.map(_chunk, [(E, p, p1, p2, kw) for p in parts])
        o = np.concatenate([x[0] for x in res])
        c = np.concatenate([x[1] for x in res])
        d = np.concatenate([x[2] for x in res])
    return b, o, c, d


def panel(ax, b, out, chi, bm=None, title=None):
    esc = out == scatter.ESCAPED
    col = out == scatter.COLLAPSED
    ax.scatter(b[esc], chi[esc], s=1.0, lw=0, color="#1f3b73", rasterized=True)
    if col.any():
        ax.scatter(b[col], np.full(int(col.sum()), -3.72), s=1.0, lw=0,
                   color="#b23a2e", rasterized=True)
    if bm is not None and b[0] <= bm <= b[-1]:
        ax.axvline(bm, color="#999", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(b[0], b[-1])
    ax.set_ylim(-4.0, 3.4)
    ax.set_ylabel(r"$\chi$")
    if title:
        ax.set_title(title, fontsize=9.5)


def main(out_path=os.path.join(HERE, "..", "figures", "step4_deflection.png")):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    N = 20_000
    fig, axes = plt.subplots(4, 1, figsize=(9.0, 11.0))

    # (a) regular slice
    E, p1, p2 = 0.1, 0.7, -2.1
    bm = locked.b_max(E)
    b, o, c, _ = scan(E, p1, p2, 0.05 * bm, 2.6 * bm, N)
    panel(axes[0], b, o, c, bm,
          r"(a) regular: $\tilde E=0.1$, $\phi^0=(0.70,-2.10)$ — "
          r"one capture interval, smooth $\chi$")

    # (b,c,d) singular slice and zooms
    E, p1, p2 = 0.3, 1.9, 1.3
    bm = locked.b_max(E)
    windows = [(0.05 * bm, 2.6 * bm), (1.4980, 1.5230), (1.5185, 1.5215)]
    labels = [r"(b) singular: $\tilde E=0.3$, $\phi^0=(1.90,1.30)$",
              r"(c) zoom $\times%.0f$" % ((2.55 * bm) / 0.025),
              r"(d) zoom $\times%.0f$" % ((2.55 * bm) / 0.003)]
    for ax, (lo, hi), lab in zip(axes[1:], windows, labels):
        b, o, c, _ = scan(E, p1, p2, lo, hi, N)
        ncap = int((o == scatter.COLLAPSED).sum())
        panel(ax, b, o, c, bm,
              lab + r"   $\tilde b\in[%.4f,%.4f]$, captured %d/%d" % (lo, hi, ncap, N))
    axes[-1].set_xlabel(r"$\tilde b$")
    fig.text(0.012, 0.5, "red band at the bottom = collapse (capture)",
             rotation=90, va="center", fontsize=8, color="#b23a2e")
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(out_path, dpi=140)
    print("wrote %s" % out_path)
    return out_path


if __name__ == "__main__":
    main()
