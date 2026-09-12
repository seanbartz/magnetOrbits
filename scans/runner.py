"""Chunked, checkpointed, resumable parameter sweeps.

A scan is a flat list of initial-condition tuples split into fixed-size chunks.  Each
finished chunk is appended to an ``.npz`` checkpoint, so a run can be killed and
restarted and will pick up where it left off.  Chunks are independent, so they can be
spread over processes.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from magdip import scatter  # noqa: E402

#: per-trajectory columns stored by a scan
FIELDS = ("outcome", "chi", "delay", "t_end", "spin_fraction", "L_end", "dE", "dJ",
          "nsteps")


def run_chunk(E, b, phi1, phi2, **kw):
    """Integrate one chunk and return the recorded columns as a dict of arrays."""
    r = scatter.run_batch(E, b, phi1, phi2, **kw)
    return {k: np.asarray(r[k]) for k in FIELDS}


class Scan(object):
    """A checkpointed sweep over broadcast initial-condition arrays."""

    def __init__(self, path, E, b, phi1, phi2, chunk=20_000, meta=None, **kw):
        self.path = path
        arrays = np.broadcast_arrays(*[np.asarray(v, dtype=float)
                                       for v in (E, b, phi1, phi2)])
        self.shape = arrays[0].shape
        self.E, self.b, self.phi1, self.phi2 = [a.ravel() for a in arrays]
        self.n = self.E.size
        self.chunk = int(chunk)
        self.kw = kw
        self.meta = dict(meta or {})

    # -- checkpointing --------------------------------------------------------
    def _load(self):
        if not os.path.exists(self.path):
            data = {k: np.full(self.n, np.nan) for k in FIELDS}
            data["outcome"] = np.full(self.n, -1, dtype=np.int16)
            data["done"] = np.zeros(self.n, dtype=bool)
            return data
        z = np.load(self.path, allow_pickle=False)
        return {k: z[k] for k in list(FIELDS) + ["done"]}

    def _save(self, data):
        tmp = self.path + ".tmp.npz"   # np.savez appends .npz unless already present
        np.savez_compressed(tmp, E=self.E, b=self.b, phi1=self.phi1, phi2=self.phi2,
                            shape=np.array(self.shape), **data,
                            **{"meta_" + k: np.array(v) for k, v in self.meta.items()})
        os.replace(tmp, self.path)

    # -- driving --------------------------------------------------------------
    def run(self, workers=1, verbose=True):
        data = self._load()
        todo = [(s, min(s + self.chunk, self.n))
                for s in range(0, self.n, self.chunk)]
        todo = [(a, z) for a, z in todo if not data["done"][a:z].all()]
        if not todo:
            if verbose:
                print("scan already complete: %s" % self.path)
            return self.results()

        t0 = time.time()
        if workers > 1:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                jobs = [pool.apply_async(run_chunk,
                                         (self.E[a:z], self.b[a:z], self.phi1[a:z],
                                          self.phi2[a:z]), self.kw)
                        for a, z in todo]
                for (a, z), job in zip(todo, jobs):
                    self._store(data, a, z, job.get())
                    self._save(data)
                    if verbose:
                        self._report(data, t0)
        else:
            for a, z in todo:
                out = run_chunk(self.E[a:z], self.b[a:z], self.phi1[a:z],
                                self.phi2[a:z], **self.kw)
                self._store(data, a, z, out)
                self._save(data)
                if verbose:
                    self._report(data, t0)
        return self.results()

    def _store(self, data, a, z, out):
        for k in FIELDS:
            data[k][a:z] = out[k]
        data["done"][a:z] = True

    def _report(self, data, t0):
        n = int(data["done"].sum())
        el = time.time() - t0
        print("  %8d/%d  %5.1f%%  %6.1fs elapsed  eta %6.1fs"
              % (n, self.n, 100.0 * n / self.n, el,
                 el * (self.n - n) / max(n, 1)), flush=True)

    def results(self):
        data = self._load()
        out = {k: data[k].reshape(self.shape) for k in FIELDS}
        out["b"] = self.b.reshape(self.shape)
        out["E"] = self.E.reshape(self.shape)
        out["phi1"] = self.phi1.reshape(self.shape)
        out["phi2"] = self.phi2.reshape(self.shape)
        out["done"] = data["done"].reshape(self.shape)
        return out


def load(path):
    """Load a finished scan checkpoint into the same dict layout as :meth:`Scan.results`."""
    z = np.load(path, allow_pickle=False)
    shape = tuple(z["shape"])
    out = {k: z[k].reshape(shape) for k in list(FIELDS) + ["done"]}
    for k in ("E", "b", "phi1", "phi2"):
        out[k] = z[k].reshape(shape)
    out["meta"] = {k[5:]: z[k] for k in z.files if k.startswith("meta_")}
    return out
