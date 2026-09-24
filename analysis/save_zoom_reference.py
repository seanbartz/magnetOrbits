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
from incomplete runs (missing, failed, or unresolved/timeout points), or from a missing or dirty
uncertainty summary. All inputs are validated before either file is written.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from physics.edges import completeness_problem, load_map   # noqa: E402

RUNS = ["strip_anti_scan800", "strip_mirror_scan800", "patch_tt_scan800", "patch_min_scan800"]


def main():
    # 1. Validate every input first; write nothing unless all of them pass.
    loaded = {}
    for name in RUNS:
        data = load_map(ROOT / "results" / name)
        if data["meta"]["git_dirty"] or data["meta"].get("git_mixed", False):
            raise SystemExit(f"{name} was run from a dirty tree or resumed with different code; "
                             "rerun it from committed code first")
        problem = completeness_problem(data)
        if problem:
            raise SystemExit(f"{name} is incomplete ({problem}); resume or rerun the sweep before storing it")
        loaded[name] = data
    summary = ROOT / "results" / "uncertainty" / "summary.json"
    if not summary.exists():
        raise SystemExit(f"{summary} not found; run analysis/uncertainty.py first")
    if json.loads(summary.read_text())["git_dirty"]:
        raise SystemExit("uncertainty summary was produced from a dirty tree; rerun analysis/uncertainty.py")

    # 2. Write both references.
    arrays = {}
    for name, data in loaded.items():
        key = name.replace("_scan800", "")
        arrays[f"{key}__outer_edge"] = data["outer_edge"]
        arrays[f"{key}__n_intervals"] = data["n_intervals"]
        arrays[f"{key}__along"] = data["along"]
        arrays[f"{key}__across"] = data["across"]
        arrays[f"{key}__meta"] = json.dumps({"grid": data["grid"], "settings": data["meta"]["settings"],
                                             "git_hash": data["meta"]["git_hash"], "run": name})
    out = ROOT / "tests" / "reference" / "zoom_maps.npz"
    np.savez_compressed(out, **arrays)
    shutil.copy(summary, ROOT / "tests" / "reference" / "uncertainty_summary.json")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) and tests/reference/uncertainty_summary.json")


if __name__ == "__main__":
    main()
