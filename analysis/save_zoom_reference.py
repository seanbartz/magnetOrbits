"""Store compact stage-8 results for the notebook and the regression tests.

Usage
-----
    python analysis/save_zoom_reference.py

Reads results/{strip_anti, strip_mirror, patch_tt, patch_min}_scan800 (the 4x-scan runs, the
better resolved of each pair) and results/uncertainty/summary.json, and writes

    tests/reference/zoom_maps.npz          b_out, interval counts and axes for each zoomed map,
                                           with each run's settings, grid and git hash
    tests/reference/uncertainty_summary.json

Refuses to write from runs recorded as git-dirty (the references must trace to committed code),
from incomplete runs (missing or failed points), or from an uncertainty summary that is dirty.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.edges import expected_points, load_map   # noqa: E402

RUNS = ["strip_anti_scan800", "strip_mirror_scan800", "patch_tt_scan800", "patch_min_scan800"]


def main():
    arrays = {}
    for name in RUNS:
        data = load_map(ROOT / "results" / name)
        if data["meta"]["git_dirty"]:
            raise SystemExit(f"{name} was run from a dirty tree; rerun it from committed code first")
        missing = expected_points(data["meta"]) - set(data["records"])
        if missing or data["errors"]:
            raise SystemExit(f"{name} is incomplete ({len(missing)} missing, {len(data['errors'])} failed); "
                             "resume the sweep before storing it")
        key = name.replace("_scan800", "")
        arrays[f"{key}__outer_edge"] = data["outer_edge"]
        arrays[f"{key}__n_intervals"] = data["n_intervals"]
        arrays[f"{key}__along"] = data["along"]
        arrays[f"{key}__across"] = data["across"]
        arrays[f"{key}__meta"] = json.dumps({"grid": data["grid"], "settings": data["meta"]["settings"],
                                             "git_hash": data["meta"]["git_hash"], "run": name})
    out = ROOT / "tests" / "reference" / "zoom_maps.npz"
    np.savez_compressed(out, **arrays)
    summary = ROOT / "results" / "uncertainty" / "summary.json"
    if json.loads(summary.read_text())["git_dirty"]:
        raise SystemExit("uncertainty summary was produced from a dirty tree; rerun analysis/uncertainty.py")
    shutil.copy(summary, ROOT / "tests" / "reference" / "uncertainty_summary.json")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) and tests/reference/uncertainty_summary.json")


if __name__ == "__main__":
    main()
