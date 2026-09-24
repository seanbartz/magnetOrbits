"""Fine structure of the capture boundary: uncertainty exponent and resolution scaling.

Place in the program
--------------------
    analysis/uncertainty.py (parallel driver) / notebook / tests
        -> fractal.uncertain_counts, fractal.fit_exponent, fractal.resolution_scaling
            -> solve.run_trajectory(stop="escape")        (the one code path)
            -> edges.capture_profile                       (for resolution scaling)

Uncertainty exponent (McDonald, Grebogi, Ott & Yorke, Physica D 17, 125 (1985); Ott & Tel,
Chaos 3, 417 (1993))
-------------------------------------------------------------------------------------------
For impact parameters b drawn uniformly in [b_lo, b_hi] (fixed orientation and energy), b is
"epsilon-uncertain" if the outcome (contact or escape) at b - eps, b or b + eps is not the
same. The uncertain fraction scales as

    f(eps) ~ eps^alpha,     D = 1 - alpha   (uncertainty dimension of the boundary set).

* A finite set of N smooth edges gives exactly f = 2 N eps / L (alpha = 1) once eps is below
  the edge spacing, with L the length the centers are drawn from (effective_length): the known
  result used to validate the method.
* alpha < 1 means the boundary is fractal: refining the b resolution keeps uncovering edges.

The same b sample is used at every eps (a common-random-numbers design: the ratios f(eps1)/f(eps2)
are less noisy). Counting errors are Poisson; alpha comes from a weighted least-squares fit of
log f against log eps over the eps values with at least `min_count` uncertain samples.
eps must stay well above the resolution of a single edge (~1e-9 relative at rtol = 1e-10,
stages 2-3), otherwise integration error, not the dynamics, decides the outcomes.

Resolution scaling (box counting)
---------------------------------
capture_profile at scan spacing db resolves N(db) edges. For a finite smooth boundary N
saturates; for a fractal one N(db) ~ db^(-D). The capture measure (total colliding length)
converges in both cases if the fractal is "fat" (positive measure), which is the case the
archived study reported for a related slice.
"""

import numpy as np

from physics.edges import OUTCOME_CODE, capture_profile
from physics.solve import run_trajectory


def outcome_codes(energy, m1, m2, b_values, settings):
    """Outcome code ('c', 'e', 't') for each b (stop='escape'); `settings` are run_trajectory kwargs."""
    return "".join(OUTCOME_CODE[run_trajectory(energy, float(b), m1, m2, stop="escape", **settings)["outcome"]]
                   for b in b_values)


def effective_length(b_lo, b_hi, epsilons):
    """Length of the interval the centers are drawn from: (b_hi - b_lo) - 2 max(eps)."""
    return (b_hi - b_lo) - 2 * max(epsilons)


def sample_points(b_lo, b_hi, epsilons, n_samples, seed):
    """Random centers b in [b_lo + eps_max, b_hi - eps_max] and the flat list of all b to evaluate.

    Returns (centers (n,), eps (k,), b_all (n * (1 + 2k),)) with b_all ordered as
    [centers, centers - eps_0, centers + eps_0, centers - eps_1, ...].
    """
    eps = np.asarray(sorted(epsilons, reverse=True), dtype=float)
    rng = np.random.default_rng(seed)
    centers = rng.uniform(b_lo + eps[0], b_hi - eps[0], size=n_samples)
    parts = [centers] + [centers + sign * e for e in eps for sign in (-1.0, 1.0)]
    return centers, eps, np.concatenate(parts)


def uncertain_counts(codes, n_samples, eps):
    """Per-eps (uncertain count, samples used, samples with a timeout) from the flat code string."""
    codes = np.frombuffer(codes.encode(), dtype="S1").reshape(1 + 2 * len(eps), n_samples)
    center = codes[0]
    rows = []
    for k in range(len(eps)):
        lo, hi = codes[1 + 2 * k], codes[2 + 2 * k]
        timeout = (center == b"t") | (lo == b"t") | (hi == b"t")
        uncertain = ((lo != center) | (hi != center)) & ~timeout
        rows.append((int(uncertain.sum()), int((~timeout).sum()), int(timeout.sum())))
    return rows


def fit_exponent(eps, counts, totals, min_count=10, eps_max=np.inf):
    """Weighted fit of log f = alpha log eps + c. Returns (alpha, alpha_err, mask used).

    Uses eps values with at least `min_count` uncertain samples and eps <= eps_max. Choose
    eps_max well below the interval length (e.g. L/30): once eps is comparable to L (or to the
    spacing of the edges) f saturates and the slope no longer measures the boundary.
    """
    eps, counts, totals = map(np.asarray, (eps, counts, totals))
    use = (counts >= min_count) & (eps <= eps_max)
    if use.sum() < 3:
        return np.nan, np.nan, use
    x = np.log(eps[use])
    y = np.log(counts[use] / totals[use])
    w = counts[use].astype(float)                 # 1 / var(log f) = n for Poisson counts
    W = np.sum(w)
    xm, ym = np.sum(w * x) / W, np.sum(w * y) / W
    sxx = np.sum(w * (x - xm) ** 2)
    alpha = np.sum(w * (x - xm) * (y - ym)) / sxx
    return float(alpha), float(1.0 / np.sqrt(sxx)), use


def resolution_scaling(energy, m1, m2, b_lo, b_hi, scan_sizes, settings, rel_tol=1e-8):
    """Edges and capture measure in [b_lo, b_hi] at each scan size (box-counting view)."""
    rows = []
    for n in scan_sizes:
        prof = capture_profile(energy, m1, m2, b_min=b_lo, b_max=b_hi, n_scan=int(n), rel_tol=rel_tol, **settings)
        rows.append({"n_scan": int(n), "db": (b_hi - b_lo) / (n - 1), "n_edges": len(prof["edges"]),
                     "n_intervals": prof["n_intervals"], "capture_measure": prof["capture_measure"],
                     "edges": [float(e) for e in prof["edges"]], "unresolved": len(prof["unresolved"])})
    return rows
