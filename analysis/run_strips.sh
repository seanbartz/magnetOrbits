#!/bin/bash
# Stage 8b/8c: production strips and their 4x scan-refinement copies (numba, r0 = 64000).
set -e
cd "$(dirname "$0")/.."   # repository root
export PYTHONPATH=.
A="--strip antiparallel --along-pi 0 1 64 --across-pi -0.078125 0.078125 81"
M="--strip mirror --along-pi 0 1 64 --across-pi -0.1875 0.09375 145"
python3 sweep.py --name strip_anti   $A --method numba
python3 sweep.py --name strip_mirror $M --method numba
python3 sweep.py --name strip_anti_scan800   $A --n-scan 800 --method numba
python3 sweep.py --name strip_mirror_scan800 $M --n-scan 800 --method numba
echo ALL_DONE
