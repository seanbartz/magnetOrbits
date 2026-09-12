"""Chunked, checkpointed, resumable parameter sweeps.

A scan is a flat list of initial-condition tuples split into fixed-size chunks.  Each
finished chunk is written as its own small shard under ``<path>.parts/``, so the cost of
checkpointing is proportional to the chunk just completed rather than to the whole scan;
the shards are consolidated into a single ``.npz`` when the sweep finishes.  A run can be
killed at any point and restarted, and will pick up where it left off.

Every checkpoint carries a fingerprint of the initial conditions and integration
settings that produced it.  Reusing an output path for a *different* sweep is refused
rather than silently merging incompatible results.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from magdip import scatter  # noqa: E402

#: per-trajectory columns stored by a scan
FIELDS = ("outcome", "chi", "delay", "t_end", "spin_fraction", "L_end", "dE", "dJ",
          "nsteps")


class CheckpointMismatch(RuntimeError):
    """Raised when an existing checkpoint was produced by a different sweep."""


def run_chunk(E, b, phi1, phi2, **kw):
    """Integrate one chunk and return the recorded columns as a dict of arrays."""
    r = scatter.run_batch(E, b, phi1, phi2, **kw)
    return {k: np.asarray(r[k]) for k in FIELDS}


def _fingerprint(E, b, phi1, phi2, shape, kw):
    """Stable hash of everything that determines the results."""
    h = hashlib.sha256()
    for a in (E, b, phi1, phi2):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    h.update(repr(tuple(shape)).encode())
    h.update(repr(sorted((str(k), repr(v)) for k, v in kw.items())).encode())
    return h.hexdigest()


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
        self.fingerprint = _fingerprint(self.E, self.b, self.phi1, self.phi2,
                                        self.shape, kw)

    # -- checkpointing --------------------------------------------------------
    @property
    def parts_dir(self):
        return self.path + ".parts"

    def _blank(self):
        data = {k: np.full(self.n, np.nan) for k in FIELDS}
        data["outcome"] = np.full(self.n, -1, dtype=np.int16)
        data["done"] = np.zeros(self.n, dtype=bool)
        return data

    def _check_fingerprint(self, stored, source):
        if str(stored) != self.fingerprint:
            raise CheckpointMismatch(
                "%s was produced by a different sweep (fingerprint %s..., expected "
                "%s...).\nThe initial conditions or integration settings have changed. "
                "Write to a new --out path, or delete the stale checkpoint."
                % (source, str(stored)[:12], self.fingerprint[:12]))

    def _load(self):
        """Reassemble progress from the consolidated file and/or the shards."""
        data = self._blank()
        if os.path.exists(self.path):
            z = np.load(self.path, allow_pickle=False)
            self._check_fingerprint(z["fingerprint"], self.path)
            for k in list(FIELDS) + ["done"]:
                data[k] = z[k]
        for shard in sorted(glob.glob(os.path.join(self.parts_dir, "chunk_*.npz"))):
            z = np.load(shard, allow_pickle=False)
            self._check_fingerprint(z["fingerprint"], shard)
            a, zz = int(z["start"]), int(z["stop"])
            for k in FIELDS:
                data[k][a:zz] = z[k]
            data["done"][a:zz] = True
        return data

    def _save_shard(self, a, z, out):
        """Write one chunk. Cost is O(chunk), not O(total)."""
        os.makedirs(self.parts_dir, exist_ok=True)
        tmp = os.path.join(self.parts_dir, "chunk_%09d_%09d.tmp.npz" % (a, z))
        np.savez(tmp, start=a, stop=z, fingerprint=self.fingerprint, **out)
        os.replace(tmp, os.path.join(self.parts_dir,
                                     "chunk_%09d_%09d.npz" % (a, z)))

    def _consolidate(self, data):
        """Write the single-file checkpoint and drop the shards."""
        tmp = self.path + ".tmp.npz"   # np.savez appends .npz unless already present
        np.savez_compressed(tmp, E=self.E, b=self.b, phi1=self.phi1, phi2=self.phi2,
                            shape=np.array(self.shape),
                            fingerprint=self.fingerprint, **data,
                            **{"meta_" + k: np.array(v) for k, v in self.meta.items()})
        os.replace(tmp, self.path)
        for shard in glob.glob(os.path.join(self.parts_dir, "chunk_*.npz")):
            os.remove(shard)
        if os.path.isdir(self.parts_dir) and not os.listdir(self.parts_dir):
            os.rmdir(self.parts_dir)

    # -- driving --------------------------------------------------------------
    def run(self, workers=1, verbose=True):
        data = self._load()
        todo = [(s, min(s + self.chunk, self.n))
                for s in range(0, self.n, self.chunk)]
        todo = [(a, z) for a, z in todo if not data["done"][a:z].all()]
        if not todo:
            if not os.path.exists(self.path):
                self._consolidate(data)
            if verbose:
                print("scan already complete: %s" % self.path)
            return self.results()

        t0 = time.time()
        if workers > 1:
            import multiprocessing as mp
            with mp.Pool(workers) as pool:
                jobs = [(a, z, pool.apply_async(
                    run_chunk, (self.E[a:z], self.b[a:z], self.phi1[a:z],
                                self.phi2[a:z]), self.kw)) for a, z in todo]
                for a, z, job in jobs:
                    out = job.get()
                    self._store(data, a, z, out)
                    self._save_shard(a, z, out)
                    if verbose:
                        self._report(data, t0)
        else:
            for a, z in todo:
                out = run_chunk(self.E[a:z], self.b[a:z], self.phi1[a:z],
                                self.phi2[a:z], **self.kw)
                self._store(data, a, z, out)
                self._save_shard(a, z, out)
                if verbose:
                    self._report(data, t0)
        self._consolidate(data)
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
    out["fingerprint"] = str(z["fingerprint"]) if "fingerprint" in z.files else None
    return out
