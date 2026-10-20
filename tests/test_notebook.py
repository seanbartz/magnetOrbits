"""The notebooks (notebooks/progress_summary.ipynb, delta_phi_maps.ipynb) must run top to bottom.

It imports everything from physics/, so an interface change that breaks it is caught here.
Executed on a temporary copy (the committed notebook keeps its outputs). ~15-30 s.
Needs the optional `notebook` dependencies (pyproject: pip install -e ".[notebook]");
skipped with a clear reason if nbconvert or ipykernel is missing. The notebooks run on a
temporary kernel for the interpreter running pytest, whatever kernel their metadata names (a
local one such as environment.yml's `magnetorbits` need not exist here).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / "notebooks" / "progress_summary.ipynb", ROOT / "notebooks" / "delta_phi_maps.ipynb"]


@pytest.mark.slow
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes(tmp_path, notebook):
    pytest.importorskip("nbconvert", reason="needs the 'notebook' extra: nbconvert")
    pytest.importorskip("ipykernel", reason="needs the 'notebook' extra: ipykernel")
    copy = tmp_path / notebook.name
    shutil.copy(notebook, copy)
    subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(tmp_path / "kernel"),
                    "--name", "magnetorbits-test"], check=True, capture_output=True)
    env = {**os.environ, "MAGNETORBITS_ROOT": str(ROOT),     # the copy lives outside the repository
           "MAGNETORBITS_NO_SAVE": "1",             # do not overwrite the committed figures
           "JUPYTER_PATH": str(tmp_path / "kernel" / "share" / "jupyter")}
    out = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                          "--ExecutePreprocessor.timeout=900", "--ExecutePreprocessor.kernel_name=magnetorbits-test",
                          "--output", str(copy), str(copy)],
                         capture_output=True, text=True, timeout=1200, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
