"""Locate the capture (collision) edge in impact parameter.

Place in the program
--------------------
    run_edge.py / sweeps / tests -> capture.find_capture_edge -> solve.run_trajectory

Definition
----------
For fixed E_tilde and initial moments, the outcome of solve.run_trajectory is a function
of b alone: "contact" (spheres touch, r = 2) or "escape" (rigorous criterion). The edge
b_c is where it changes from contact (b < b_c) to escape (b > b_c).

Method and its assumption
-------------------------
Bisection on the outcome inside a bracket [b_lo, b_hi] with outcome(b_lo) = contact and
outcome(b_hi) = escape. Bisection returns *a* boundary point in the bracket; it is THE
edge only if the contact set is a single interval there. That holds for the central
-kappa/r^3 force (stage 3 check against physics/central.py) but need not hold once the
moments rotate (the archived study found fractal capture sets in parts of parameter
space). For those cases use a scan first and bisect on each sign change.

Runs use stop="escape" (terminate at the rigorous escape criterion): the deflection is
not needed, and this avoids integrating the outgoing leg to r_out.

Accuracy: the outcome near the edge is decided by an O(rtol) energy error, so the edge is
located to roughly b_c * rtol / E; the default tolerance 1e-9 (relative) sits above that
for rtol = 1e-10 and E >~ 0.01.
"""

import numpy as np

from physics.solve import run_trajectory


def outcome_at(energy, impact_parameter, m1, m2, **trajectory_kwargs):
    """Outcome string of one run with stop="escape" (a timeout raises: undecided)."""
    res = run_trajectory(energy, impact_parameter, m1, m2, stop="escape", **trajectory_kwargs)
    if res["outcome"] == "timeout":
        raise RuntimeError(f"timeout at E={energy}, b={impact_parameter}: edge undecided")
    return res["outcome"]


def find_capture_edge(energy, m1, m2, b_lo, b_hi, rel_tol=1e-9, max_iter=100, **trajectory_kwargs):
    """Bisect for the contact/escape boundary in [b_lo, b_hi].

    Returns dict(edge, b_contact, b_escape, n_runs): the final bracket
    b_contact < edge < b_escape has b_escape - b_contact < rel_tol * edge.
    """
    n_runs = 0

    def outcome(b):
        nonlocal n_runs
        n_runs += 1
        return outcome_at(energy, b, m1, m2, **trajectory_kwargs)

    if outcome(b_lo) != "contact":
        raise ValueError(f"b_lo = {b_lo} does not collide")
    if outcome(b_hi) != "escape":
        raise ValueError(f"b_hi = {b_hi} does not escape")

    b_contact, b_escape = b_lo, b_hi
    for _ in range(max_iter):
        if b_escape - b_contact < rel_tol * 0.5 * (b_contact + b_escape):
            break
        mid = 0.5 * (b_contact + b_escape)
        if outcome(mid) == "contact":
            b_contact = mid
        else:
            b_escape = mid
    return {"edge": 0.5 * (b_contact + b_escape), "b_contact": b_contact,
            "b_escape": b_escape, "n_runs": n_runs}
