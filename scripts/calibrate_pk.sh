#!/usr/bin/env bash
# Calibration of the pseudoknot loop term (M4 = M3 + pk_mode=loop) on the
# calibration fold, same chains / limits as the pk_beta sweep.
# Usage (WSL):  bash scripts/calibrate_pk.sh   [GRID="6.0:0.1 9.0:0.1"]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/micromamba/envs/mrna-sqd/bin/python}
for g in ${GRID:-6.0:0.1 6.0:0.25 9.0:0.1 9.0:0.25 12.0:0.1 12.0:0.25}; do
  a=${g%%:*}; u=${g##*:}
  "$PY" scripts/run_models.py --fold calibration --models M4 --max-len 150 \
      --time-limit ${TL:-60} --workers 4 --procs 8 \
      --out results/calib_pk.jsonl \
      --set M4.pk_init=$a --set M4.pk_unp=$u --tag "_i${a}_u${u}"
done
