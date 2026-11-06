#!/usr/bin/env bash
# Held-out test fold: every model, one after another, same CPU budget.
# Usage (WSL):  bash scripts/run_test_fold.sh   [MODELS_TO_RUN="M3 M0"]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/micromamba/envs/mrna-sqd/bin/python}
for m in ${MODELS_TO_RUN:-M3 M0 M1 FOX22}; do
  "$PY" scripts/run_models.py --fold test --models "$m" --max-len 150 \
      --time-limit 60 --workers 4 --procs 8 --out results/test_fold.jsonl \
      > "results/test_${m}.log" 2>&1
done
