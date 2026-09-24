"""Vectorised integrators for large ensembles of trajectories.

Two schemes:

* :func:`integrate_batch` -- explicit Dormand-Prince 5(4) with *per-trajectory*
  adaptive step size, run in lock-step over a batch, with bisection-refined terminal
  events.  This is the workhorse for parameter scans.
* :func:`gauss_legendre_step` / :func:`integrate_gl` -- fixed-step Gauss-Legendre
  implicit Runge-Kutta (2-stage order 4, 3-stage order 6).  Symplectic; used as an
  independent cross-check on long-lived trajectories, where bounded rather than
  secular energy error matters.

State arrays are ``(D, N)``: state-first, one column per trajectory.
"""
from __future__ import annotations

import numpy as np

# --- Dormand-Prince 5(4) tableau -------------------------------------------------
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
]
_B5 = np.array([35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0])
_B4 = np.array([5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200,
                187 / 2100, 1 / 40])
_E = _B5 - _B4
_ORDER = 5

# Hairer's free 4th-order dense-output coefficients for DOPRI5
_D = np.array([-12715105075 / 11282082432, 0.0, 87487479700 / 32700410799,
               -10690763975 / 1880347072, 701980252875 / 199316789632,
               -1453857185 / 822651844, 69997945 / 29380423])


def _dp_stages(f, t, y, h):
    """Seven Dormand-Prince stages; returns ``(k, y5, err)``."""
    k = [f(t, y)]
    for i in range(1, 7):
        acc = np.zeros_like(y)
        for j, a in enumerate(_A[i]):
            if a:
                acc += a * k[j]
        k.append(f(t + _C[i] * h, y + h * acc))
    y5 = y + h * sum(b * kk for b, kk in zip(_B5, k) if b)
    err = h * sum(e * kk for e, kk in zip(_E, k) if e)
    return k, y5, err


def _dp_interp(y, y_new, h, k, theta):
    """Dense output of the accepted DOPRI5 step at fraction ``theta`` in ``[0, 1]``.

    Fourth-order accurate and free -- it reuses the seven stages, so locating an event
    costs no further right-hand-side evaluations (and therefore cannot disturb any
    per-column data the RHS closes over).
    """
    ydiff = y_new - y
    c3 = h * k[0] - ydiff
    c4 = ydiff - h * k[6] - c3
    c5 = h * sum(d * kk for d, kk in zip(_D, k) if d)
    th = theta
    om = 1.0 - theta
    return y + th * (ydiff + om * (c3 + th * (c4 + om * c5)))


def _err_norm(err, y, y_new, rtol, atol):
    """Per-column RMS error norm."""
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    return np.sqrt(np.mean((err / scale) ** 2, axis=0))


class Event(object):
    """A terminal event ``g(t, y) = 0`` crossed in a given direction."""

    def __init__(self, g, direction, code, name=""):
        self.g = g
        self.direction = direction
        self.code = int(code)
        self.name = name or "event%d" % code


def integrate_batch(f, y0, t_max, events, rtol=1e-10, atol=1e-12,
                    h0=None, h_min=1e-14, max_steps=2_000_000, safety=0.9,
                    fac_min=0.2, fac_max=8.0, compact_every=64,
                    timeout_code=3, progress=None, rebind=None):
    """Integrate a batch of trajectories to their first terminal event.

    Parameters
    ----------
    f : callable ``f(t, y)`` acting on ``(D, N)`` arrays, elementwise in the columns.
    y0 : ``(D, N)`` initial states.
    t_max : scalar or ``(N,)`` integration horizon.
    events : sequence of :class:`Event`.  Earlier entries win ties.
    rebind : optional ``rebind(keep) -> (f, events)``.  Finished trajectories are
        dropped from the working arrays as they complete; if ``f`` or the event
        functions close over per-trajectory data (e.g. a per-column reference energy),
        supply this hook so that data can be sliced by the same boolean mask ``keep``,
        which is taken over the *currently active* columns.

    Returns
    -------
    dict with ``y`` (final states), ``t`` (end times), ``outcome`` (event code, or
    ``timeout_code``), ``nsteps`` and ``nrejected`` per trajectory.
    """
    y0 = np.asarray(y0, dtype=float)
    D, N = y0.shape
    t_max = np.broadcast_to(np.asarray(t_max, dtype=float), (N,)).copy()

    y = y0.copy()
    t = np.zeros(N)
    idx = np.arange(N)                      # active -> original column
    out_y = np.empty_like(y0)
    out_t = np.zeros(N)
    outcome = np.full(N, timeout_code, dtype=np.int8)
    nsteps = np.zeros(N, dtype=np.int64)
    nrej = np.zeros(N, dtype=np.int64)

    if h0 is None:
        f0 = f(0.0, y)
        d0 = np.sqrt(np.mean((y / (atol + rtol * np.abs(y))) ** 2, axis=0))
        d1 = np.sqrt(np.mean((f0 / (atol + rtol * np.abs(y))) ** 2, axis=0))
        h = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / np.maximum(d1, 1e-300))
    else:
        h = np.broadcast_to(np.asarray(h0, dtype=float), (N,)).copy()
    h = np.minimum(h, np.maximum(t_max, h_min))

    gvals = [ev.g(t, y) for ev in events]

    step = 0
    while idx.size and step < max_steps:
        step += 1
        hh = np.minimum(h, t_max - t)
        k, y_new, err = _dp_stages(f, t, y, hh)
        e = _err_norm(err, y, y_new, rtol, atol)
        bad = ~np.isfinite(e)
        e = np.where(bad, np.inf, e)
        ok = (e <= 1.0) | (hh <= h_min)

        # step-size control
        with np.errstate(divide="ignore", over="ignore"):
            fac = safety * np.where(e > 0, e ** (-1.0 / _ORDER), fac_max)
        fac = np.clip(fac, fac_min, fac_max)
        fac = np.where(ok, fac, np.minimum(fac, 1.0))
        h = np.maximum(hh * fac, h_min)

        nrej[idx[~ok]] += 1
        if not ok.any():
            continue

        t_try = t + hh
        # --- terminal events on accepted steps ---------------------------------
        trig = np.zeros(idx.size, dtype=bool)
        which = np.full(idx.size, -1, dtype=np.int8)
        t_hit = t_try.copy()
        y_hit = y_new.copy()
        for ei, ev in enumerate(events):
            g_old, g_new = gvals[ei], ev.g(t_try, y_new)
            gvals[ei] = g_new
            cross = ok & ~trig & _crossed(g_old, g_new, ev.direction)
            if cross.any():
                th, yh = _locate(t, y, y_new, hh, k, ev, g_old, cross)
                t_hit = np.where(cross, th, t_hit)
                y_hit = np.where(cross, yh, y_hit)
                which = np.where(cross, ev.code, which)
                trig |= cross

        y = np.where(ok, np.where(trig, y_hit, y_new), y)
        t = np.where(ok, np.where(trig, t_hit, t_try), t)
        for ei, ev in enumerate(events):     # resync after bisection/rejection
            gvals[ei] = ev.g(t, y)
        nsteps[idx[ok]] += 1

        done = trig | (ok & (t >= t_max - 1e-300))
        if done.any():
            out_y[:, idx[done]] = y[:, done]
            out_t[idx[done]] = t[done]
            outcome[idx[done]] = np.where(trig[done], which[done], timeout_code)
            keep = ~done
            idx = idx[keep]
            y, t, h, t_max = y[:, keep], t[keep], h[keep], t_max[keep]
            if rebind is not None:
                f, events = rebind(keep)
            gvals = [ev.g(t, y) for ev in events]
        if progress is not None and compact_every and step % compact_every == 0:
            progress(step, idx.size, N)

    if idx.size:                             # ran out of steps
        out_y[:, idx] = y
        out_t[idx] = t
        outcome[idx] = timeout_code

    return {"y": out_y, "t": out_t, "outcome": outcome,
            "nsteps": nsteps, "nrejected": nrej}


def _crossed(g_old, g_new, direction):
    if direction > 0:
        return (g_old < 0.0) & (g_new >= 0.0)
    if direction < 0:
        return (g_old > 0.0) & (g_new <= 0.0)
    return np.sign(g_old) != np.sign(g_new)


def _locate(t, y, y_new, h, k, ev, g_old, mask, iters=60):
    """Bisect the event function on the step's dense output."""
    g0 = g_old
    lo = np.zeros(t.shape)
    hi = np.ones(t.shape)
    t_hit = t + h
    y_hit = y_new.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        y_mid = _dp_interp(y, y_new, h, k, mid)
        g_mid = ev.g(t + h * mid, y_mid)
        before = (np.sign(g_mid) == np.sign(g0)) & (g_mid != 0.0)
        lo = np.where(before, mid, lo)
        hi = np.where(before, hi, mid)
        t_hit = np.where(before, t_hit, t + h * mid)
        y_hit = np.where(before, y_hit, y_mid)
    return np.where(mask, t_hit, t + h), np.where(mask, y_hit, y_new)


# --- Gauss-Legendre implicit RK (symplectic) -------------------------------------
_S3 = np.sqrt(3.0)
_S15 = np.sqrt(15.0)
GL_TABLEAUX = {
    2: (np.array([[0.25, 0.25 - _S3 / 6], [0.25 + _S3 / 6, 0.25]]),
        np.array([0.5, 0.5]),
        np.array([0.5 - _S3 / 6, 0.5 + _S3 / 6])),
    3: (np.array([[5 / 36, 2 / 9 - _S15 / 15, 5 / 36 - _S15 / 30],
                  [5 / 36 + _S15 / 24, 2 / 9, 5 / 36 - _S15 / 24],
                  [5 / 36 + _S15 / 30, 2 / 9 + _S15 / 15, 5 / 36]]),
        np.array([5 / 18, 4 / 9, 5 / 18]),
        np.array([0.5 - _S15 / 10, 0.5, 0.5 + _S15 / 10])),
}


class StageSolveError(RuntimeError):
    """The implicit stage equations did not converge within ``max_iter``."""


def gauss_legendre_step(f, t, y, h, stages=2, tol=1e-13, max_iter=100, strict=True):
    """One symplectic Gauss-Legendre step, solved by fixed-point iteration.

    The scheme is symplectic and of order ``2*stages`` *only* if the implicit stage
    equations are actually solved, so a fixed-point iteration that runs out of
    iterations raises :class:`StageSolveError` rather than silently returning the last
    iterate as though it were a valid step.  Pass ``strict=False`` to accept it anyway
    (and lose both guarantees).  Fixed-point iteration converges geometrically for
    ``h`` below roughly the inverse Lipschitz constant; if it fails, reduce ``h``.
    """
    A, b, c = GL_TABLEAUX[stages]
    k = [f(t, y).copy() for _ in range(stages)]
    delta = np.inf
    for _ in range(max_iter):
        k_new = []
        for i in range(stages):
            acc = np.zeros_like(y)
            for j in range(stages):
                acc = acc + A[i, j] * k[j]
            k_new.append(f(t + c[i] * h, y + h * acc))
        delta = max(float(np.max(np.abs(kn - ko))) for kn, ko in zip(k_new, k))
        k = k_new
        if delta < tol:
            break
    else:
        if strict:
            raise StageSolveError(
                "Gauss-Legendre stage equations did not converge in %d iterations "
                "at h=%g (residual %.3e > tol %.3e); reduce the step size."
                % (max_iter, h, delta, tol))
    return y + h * sum(bi * ki for bi, ki in zip(b, k))


def integrate_gl(f, y0, h, nsteps, stages=2, t0=0.0, record_every=0):
    """Fixed-step symplectic integration; optionally records a trajectory."""
    y = np.asarray(y0, dtype=float).copy()
    t = float(t0)
    rec_t, rec_y = ([t], [y.copy()]) if record_every else ([], [])
    for n in range(1, nsteps + 1):
        y = gauss_legendre_step(f, t, y, h, stages=stages)
        t += h
        if record_every and n % record_every == 0:
            rec_t.append(t)
            rec_y.append(y.copy())
    if record_every:
        return y, t, np.array(rec_t), np.array(rec_y).T
    return y, t
