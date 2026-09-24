#!/bin/bash
# Stage 8B: 2D patch around tail-to-tail (phi1, phi2) = (pi, 0), steps pi/512 in both angles,
# production scan and its 4x scan-refinement copy (numba, r0 = 64000).
set -e
cd "$(dirname "$0")/.."   # repository root
export PYTHONPATH=.
B="--strip box --along-pi 0.90625 1.095703125 97 --across-pi -0.09375 0.09375 97"
python3 sweep.py --name patch_tt $B --method numba
python3 sweep.py --name patch_tt_scan800 $B --n-scan 800 --method numba
echo ALL_DONE
# Finer patch around the deep minimum (0.974609375 pi, -0.025390625 pi) = (7984, -208)/8192 pi,
# steps pi/8192, 65 x 65 points, with its 4x scan copy.
F="--strip box --along-pi 0.9736328125 0.9815673828125 65 --across-pi -0.029296875 -0.021484375 65"
python3 sweep.py --name patch_min $F --method numba
python3 sweep.py --name patch_min_scan800 $F --n-scan 800 --method numba
echo FINE_DONE
