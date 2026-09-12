import os
import sys

_ROOT = os.path.dirname(__file__)
for _sub in ("src", "scans"):
    sys.path.insert(0, os.path.join(_ROOT, _sub))
