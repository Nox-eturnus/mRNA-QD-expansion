#!/usr/bin/env bash
# Calibration of the turner_direct model (M3) on the calibration fold:
# sweep the pseudoknot weight pk_beta.  Objective: mean MCC over all
# calibration chains.  Usage (WSL):  bash scripts/calibrate.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/calib_sweep.jsonl
RUN="${PY:-$HOME/micromamba/envs/mrna-sqd/bin/python} scripts/run_models.py --fold calibration --models M3 --max-len 150 \
     --time-limit ${TL:-60} --workers 4 --procs 8 --out $OUT"

# c_far and ml_sibling stay at 0: non-zero values couple almost every
# nested / side-by-side stem pair, J turns dense and CP-SAT stops proving
# optimality above ~40 nt (c_far = -1: 0 % optimal at 60-80 nt).  Only the
# pseudoknot weight is calibrated.
for b in ${BETAS:--1.0 -0.5 0.0 0.5 1.0 2.0 3.0}; do
  $RUN --set M3.c_far=0.0 --set M3.pk_beta=$b --tag "_b${b}"
done
