"""Orientation maps in (phi1, dphi) and (sigma, dphi) coordinates.

    dphi  = phi2 - phi1, wrapped to [0, 2 pi)   (antiparallel at the center, parallel at the edges)
    sigma = phi1 + phi2, wrapped to [0, 2 pi)   (the mirror line phi1 + phi2 = pi is sigma = pi)

No new physics: the maps are the stored ones (tests/reference/map_torus32.npz, zoom_maps.npz),
re-indexed. Used by notebooks/delta_phi_maps.ipynb.

(phi1, dphi): exact index shift. Every stored map samples dphi on one uniform lattice (the torus
because its phi1 and phi2 steps are equal; the strips and patches because their steps are commensurate),
so each sample moves to an integer row of a rectangular (phi1, dphi) array. `delta_grid` checks that
the shift is an integer to 1e-9 of a step and raises otherwise; nothing is interpolated.

(sigma, dphi): the torus lattice maps to a checkerboard (i + j and j - i have the same parity), so there
is no rectangular array. `cell_mesh` instead draws each sample's own cell, mapped by the linear change of
variables (a diamond or a parallelogram). Again nothing is interpolated.

Symmetries in these coordinates (b > 0 outer edge; see notebooks/progress_summary.ipynb, Section 8):
    flip both moments  (phi1, phi2) -> (phi1 + pi, phi2 + pi):  (phi1, dphi) -> (phi1 + pi, dphi);
                                                                (sigma, dphi) unchanged
    exchange spheres   (phi1, phi2) -> (phi2, phi1):            (sigma, dphi) -> (sigma, -dphi)
    b -> -b            (phi1, phi2) -> (pi - phi1, pi - phi2):  with exchange, (sigma, dphi) -> (2 pi - sigma, dphi)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics.edges import strip_angles   # noqa: E402

TWO_PI = 2 * np.pi


def wrap(x, lo=0.0):
    """x mod 2 pi, into [lo, lo + 2 pi)."""
    return (np.asarray(x) - lo) % TWO_PI + lo


def sample_angles(kind, along, across):
    """(phi1, phi2) at every sample, shape (len(along), len(across)).

    kind "torus": along = across = phi grid, values[i, j] at (phi[i], phi[j]).
    Otherwise a strip line of physics.edges.strip_angles ("antiparallel", "mirror", "box").
    """
    s, d = np.meshgrid(along, across, indexing="ij")
    if kind == "torus":
        return s, d
    return strip_angles(kind, s, d)


def delta_grid(values, phi1, phi2, lo=0.0):
    """Exact (phi1, dphi) array: values re-indexed by an integer shift of each phi1 row.

    values, phi1, phi2: shape (n_along, n_across), phi1 constant along each row (true for every
    stored map). Returns (array, phi1_axis, dphi_axis, step) with array[i, k] the sample at
    (phi1_axis[i], dphi_axis[k]); unsampled cells are NaN. dphi is wrapped to [lo, lo + 2 pi).
    """
    if not np.allclose(phi1, phi1[:, :1]):
        raise ValueError("phi1 must be constant along each row")
    dphi = phi2 - phi1
    step = np.min(np.abs(np.diff(dphi, axis=1)))
    n_circle = TWO_PI / step
    if abs(n_circle - round(n_circle)) > 1e-9 * n_circle:
        raise ValueError(f"dphi step {step} does not divide 2 pi; an index shift is not exact")
    n_circle = int(round(n_circle))
    step = TWO_PI / n_circle                              # exact lattice step (the estimate carries rounding)
    offset = (dphi.min() - lo) % step                     # lattice offset from lo (0 for every stored map)
    if offset > step - 1e-9 * step:
        offset = 0.0
    pos = (wrap(dphi, lo) - lo - offset) / step
    if np.max(np.abs(pos - np.round(pos))) > 1e-9:
        raise ValueError("dphi samples are not on one lattice; an index shift is not exact")
    rows = np.round(pos).astype(int) % n_circle           # row of each sample, counted from lo
    if any(len(np.unique(r)) != len(r) for r in rows):
        raise ValueError("two samples of one phi1 row land on the same dphi cell")
    k0, k1 = rows.min(), rows.max()                       # a band that wraps keeps the full circle
    out = np.full((values.shape[0], k1 - k0 + 1), np.nan)
    np.put_along_axis(out, rows - k0, values, axis=1)
    dphi_axis = lo + offset + step * np.arange(k0, k1 + 1)
    return out, phi1[:, 0], dphi_axis, step


def axis_edges(x):
    """Cell edges of a uniform axis (midpoints, extended by half a step at both ends)."""
    x = np.asarray(x, float)
    h = x[1] - x[0] if len(x) > 1 else 1.0
    return np.concatenate([x - h / 2, [x[-1] + h / 2]])


def extent(x, y):
    """imshow extent for uniform axes x (columns) and y (rows)."""
    ex, ey = axis_edges(x), axis_edges(y)
    return (ex[0], ex[-1], ey[0], ey[-1])


def cell_mesh(kind, along, across, coords):
    """Corner arrays (X, Y) of every sample's cell in plot coordinates, for pcolormesh.

    The cells are the rectangles of the stored (along, across) grid, mapped by the linear change of
    variables `coords(phi1, phi2) -> (x, y)` (not wrapped; see `tiled_mesh`).
    """
    p1, p2 = sample_angles(kind, axis_edges(along), axis_edges(across))
    return coords(p1, p2)


def tiled_mesh(ax, kind, along, across, values, coords, xlim, ylim, **kw):
    """pcolormesh of the mapped cells, repeated by multiples of 2 pi in x and y to fill (xlim, ylim)."""
    X, Y = cell_mesh(kind, along, across, coords)
    im = None
    shifts = range(-3, 4)
    for a in shifts:
        for b in shifts:
            Xs, Ys = X + a * TWO_PI, Y + b * TWO_PI
            if Xs.max() < xlim[0] or Xs.min() > xlim[1] or Ys.max() < ylim[0] or Ys.min() > ylim[1]:
                continue
            im = ax.pcolormesh(Xs, Ys, values, shading="flat", rasterized=True, **kw)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    return im


# Coordinate systems: (phi1, phi2) -> plot (x, y), unwrapped.
COORDS = {
    "phi": lambda p1, p2: (p1, p2),
    "delta": lambda p1, p2: (p1, p2 - p1),
    "sumdiff": lambda p1, p2: (p1 + p2, p2 - p1),
    "anti_strip": lambda p1, p2: (p2 - p1 - np.pi, p1),       # original strip axes: (d, s)
    "mirror_strip": lambda p1, p2: (p1 + p2 - np.pi, p1),
    "box": lambda p1, p2: (p2, p1),                           # original patch axes: (phi2, phi1)
}

# The three reference families as curves phi2(phi1).
LINES = {
    "mirror": (lambda t: np.pi - t, dict(ls=":", lw=2.2)),
    "antiparallel": (lambda t: t + np.pi, dict(ls="--", lw=1.6)),
    "parallel": (lambda t: t, dict(ls="-.", lw=1.6)),
}


def overlay_lines(ax, coords, color="w", which=("mirror", "antiparallel", "parallel")):
    """Draw the mirror, antiparallel and parallel lines in plot coordinates `coords`.

    Each line is sampled densely in phi1, mapped, and wrapped into a 2 pi window starting just
    below the current axis limits (breaks are inserted where the wrap jumps), so it shows up in
    every periodic copy that intersects the axes. Call after the axis limits are set.
    """
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    t = np.linspace(-2 * TWO_PI, 2 * TWO_PI, 400001)
    for name in which:
        f, style = LINES[name]
        x, y = coords(t, f(t))
        for kx in range(-1, 2):          # wrap windows shifted by whole periods cover views wider than 2 pi
            for ky in range(-1, 2):
                xw = wrap(x, x0 - 1e-9) + kx * TWO_PI
                yw = wrap(y, y0 - 1e-9) + ky * TWO_PI
                jump = np.concatenate([[False], (np.abs(np.diff(xw)) > np.pi) | (np.abs(np.diff(yw)) > np.pi)])
                xw, yw = xw.astype(float), yw.astype(float)
                xw[jump], yw[jump] = np.nan, np.nan
                inside = (xw >= x0) & (xw <= x1) & (yw >= y0) & (yw <= y1)
                if inside.any():
                    ax.plot(xw, yw, color=color, **style)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)


def line_legend(fig, color="k", **kw):
    """Legend below the panels naming the three line styles. Needs layout="constrained"; call it
    after the axes are labeled."""
    import matplotlib
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy
    handles = [Line2D([], [], color=color, **style) for _, style in LINES.values()]
    labels = [r"mirror $\phi_1+\phi_2=\pi$", r"antiparallel $\Delta\phi=\pi$", r"parallel $\Delta\phi=0$"]
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) >= (3, 7):
        return fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, **kw)
    # matplotlib < 3.7 has no "outside" placement, and its constrained layout ignores figure legends.
    # Hang the legend under the bottom-center panel instead, below its tick and axis labels, so the
    # layout makes room for it as part of that axes.
    panels = [ax for ax in fig.axes if getattr(ax, "get_subplotspec", lambda: None)() is not None]
    bottom = max(ax.get_subplotspec().rowspan.stop for ax in panels)
    row = [ax for ax in panels if ax.get_subplotspec().rowspan.stop == bottom]
    ax = min(row, key=lambda a: abs(a.get_position().x0 + a.get_position().width / 2 - 0.5))
    below = ax.bbox.y0 - ax.xaxis.get_tightbbox(fig.canvas.get_renderer()).y0     # pixels
    anchor = offset_copy(ax.transAxes, fig=fig, y=-below * 72 / fig.dpi, units="points")
    return ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0), bbox_transform=anchor,
                     ncol=3, frameon=False, **kw)


def pi_label(k, den):
    """Label for k * pi / den, reduced."""
    from math import gcd
    if k == 0:
        return "0"
    g = gcd(abs(k), den)
    k, den = k // g, den // g
    sign = "-" if k < 0 else ""
    num = "" if abs(k) == 1 else str(abs(k))
    return fr"${sign}{num}\pi$" if den == 1 else fr"${sign}{num}\pi/{den}$"


def pi_ticks(ax, axis="x", max_ticks=5, min_den=2, max_den=64):
    """Ticks at multiples of pi/den, den = 2, 4, 8, ... chosen to give at most max_ticks ticks.

    Full-range axes get multiples of pi/2. Narrow zooms that need den > max_den get decimal
    multiples of pi (e.g. 0.9750 pi) instead of long fractions.
    """
    lo, hi = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
    den = min_den
    while True:
        ks = np.arange(np.ceil(lo / (np.pi / den) - 1e-9), np.floor(hi / (np.pi / den) + 1e-9) + 1).astype(int)
        if len(ks) >= 2 or den >= 4096:
            break
        den *= 2
    while len(ks) > max_ticks and den > min_den:
        den //= 2
        ks = np.arange(np.ceil(lo / (np.pi / den) - 1e-9), np.floor(hi / (np.pi / den) + 1e-9) + 1).astype(int)
    if len(ks) > max_ticks:
        ks = ks[:: int(np.ceil(len(ks) / max_ticks))]
    vals = ks * np.pi / den
    labels = [pi_label(k, den) for k in ks] if den <= max_den else [fr"${k / den:.4f}\pi$" for k in ks]
    (ax.set_xticks if axis == "x" else ax.set_yticks)(vals)
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)(labels)


# Reference orientations of the progress notebook (Section 8).
NAMED = {"HT": (0.0, 0.0), "HH": (0.0, np.pi), "P": (np.pi / 2, np.pi / 2), "A": (np.pi / 2, 3 * np.pi / 2)}


def mark_named(ax, coords, color="w", fontsize=15):
    """Circle and label HT, HH, P, A at every periodic copy inside the axes."""
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    for name, (p1, p2) in NAMED.items():
        x, y = coords(np.float64(p1), np.float64(p2))
        for xs in wrap(x, x0 - 1e-9) + TWO_PI * np.arange(-1, 2):
            for ys in wrap(y, y0 - 1e-9) + TWO_PI * np.arange(-1, 2):
                if x0 - 1e-9 <= xs <= x1 + 1e-9 and y0 - 1e-9 <= ys <= y1 + 1e-9:
                    ax.plot(xs, ys, "o", ms=13, mfc="none", mec=color, mew=2, clip_on=False)
                    ax.text(xs + 0.12, ys + 0.12, name, color=color, fontsize=fontsize, fontweight="bold")


# ---------------------------------------------------------------------------------------------
# One map panel in one of three views.

VIEWS = ("orig", "delta", "sumdiff")
ORIG = {   # kind -> (coords name, x label, y label); x = columns of the stored array's natural layout
    "torus": ("phi", r"$\phi_1$", r"$\phi_2$"),
    "antiparallel": ("anti_strip", r"$d$   ($\phi_2=\phi_1+\pi+d$)", r"$s=\phi_1$"),
    "mirror": ("mirror_strip", r"$d$   ($\phi_2=\pi-\phi_1+d$)", r"$s=\phi_1$"),
    "box": ("box", r"$\phi_2$", r"$\phi_1$"),
}
DPHI_LABEL = r"$\Delta\phi=\phi_2-\phi_1$"
SUM_LABEL = r"$\phi_1+\phi_2$"


def _window(lo, hi):
    """Plot window for an unwrapped coordinate range: the range itself, or one full period."""
    if hi - lo >= TWO_PI - 1e-9:
        return (0.0, TWO_PI)
    shift = TWO_PI * np.floor(0.5 * (lo + hi) / TWO_PI)
    return (lo - shift, hi - shift)


def draw_map(ax, kind, along, across, values, view, named=None, line_color="w", **imkw):
    """Draw a stored map in view "orig", "delta" (phi1, dphi) or "sumdiff" (phi1+phi2, dphi).

    kind: "torus" (values[i, j] at phi1 = along[i], phi2 = across[j]) or a strip line
    ("antiparallel", "mirror", "box"; values[i, j] at strip_angles(kind, along[i], across[j])).
    Every view shows each stored sample as one cell; nothing is interpolated. Overlays the mirror,
    antiparallel and parallel lines, and the reference orientations if `named` (default: torus only).
    Returns the image artist (for a colorbar).
    """
    values = np.asarray(values, float)
    if view == "orig":
        cname, xlabel, ylabel = ORIG[kind]
        if kind == "torus":
            im = ax.imshow(values.T, origin="lower", extent=extent(along, across), interpolation="nearest", **imkw)
        else:
            im = ax.imshow(values, origin="lower", extent=extent(across, along), interpolation="nearest",
                           aspect="auto", **imkw)
    elif view == "delta":
        cname, xlabel, ylabel = "delta", r"$\phi_1$", DPHI_LABEL
        p1, p2 = sample_angles(kind, along, across)
        grid, x, y, _ = delta_grid(values, p1, p2)
        im = ax.imshow(grid.T, origin="lower", extent=extent(x, y), interpolation="nearest",
                       aspect="equal" if kind == "torus" else "auto", **imkw)
    elif view == "sumdiff":
        cname, xlabel, ylabel = "sumdiff", SUM_LABEL, DPHI_LABEL
        X, Y = cell_mesh(kind, along, across, COORDS[cname])
        xlim, ylim = _window(X.min(), X.max()), _window(Y.min(), Y.max())
        if kind == "torus":
            xlim = ylim = (0.0, TWO_PI)
        im = tiled_mesh(ax, kind, along, across, values, COORDS[cname], xlim, ylim, **imkw)
        ax.set_aspect("equal" if kind == "torus" else "auto")
    else:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    overlay_lines(ax, COORDS[cname], color=line_color)
    if named if named is not None else kind == "torus":
        mark_named(ax, COORDS[cname], color=line_color)
    pi_ticks(ax, "x"); pi_ticks(ax, "y")
    return im
