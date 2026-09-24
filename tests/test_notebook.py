"""The progress notebook (notebooks/progress_summary.ipynb) must run top to bottom.

It imports everything from physics/, so an interface change that breaks it is caught here.
Executed on a temporary copy (the committed notebook keeps its outputs). ~15-30 s.
Needs the optional `notebook` dependencies (pyproject: pip install -e ".[notebook]");
skipped with a clear reason if nbconvert or ipykernel is missing.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "progress_summary.ipynb"


@pytest.mark.slow
def test_progress_notebook_executes(tmp_path):
    pytest.importorskip("nbconvert", reason="needs the 'notebook' extra: nbconvert")
    pytest.importorskip("ipykernel", reason="needs the 'notebook' extra: ipykernel (python3 kernel)")
    copy = tmp_path / NOTEBOOK.name
    shutil.copy(NOTEBOOK, copy)
    env = {**os.environ, "MAGNETORBITS_ROOT": str(ROOT)}     # the copy lives outside the repository
    out = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                          "--ExecutePreprocessor.timeout=900", "--output", str(copy), str(copy)],
                         capture_output=True, text=True, timeout=1200, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
