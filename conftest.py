# Makes the top-level `physics` package importable from tests without installing.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
