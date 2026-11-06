#!/usr/bin/env bash
# CP-SAT top-K (K=10) for one model on both folds; first solve 60 s, then 15 s.
# Re-ranking happens offline in scripts/rerank.py.
# Usage (WSL):  MODEL=M4 bash scripts/topk.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/micromamba/envs/mrna-sqd/bin/python}
M=${MODEL:-M4}
for f in calibration test; do
  "$PY" scripts/run_models.py --fold "$f" --models "$M" --max-len 150 \
      --time-limit 60 --topk 10 --topk-time-limit 15 --workers 4 --procs 8 \
      --tag _topk --out results/topk.jsonl > "results/topk_${M}_${f}.log" 2>&1
done
