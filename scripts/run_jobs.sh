#!/usr/bin/env bash
# Run a list of run_quantum.py configurations, one per line of a job file:
#   <method> <tag> [extra run_quantum.py args...]
# Lines starting with # are skipped.  Usage (WSL):
#   SET=qdev OUT=results/qdev.jsonl bash scripts/run_jobs.sh jobs.txt
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/micromamba/envs/mrna-sqd/bin/python}
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-8}
export JAX_PLATFORMS=cpu
JOBS=$1
while read -r method tag extra; do
  [[ -z "$method" || "$method" == \#* ]] && continue
  # shellcheck disable=SC2086
  "$PY" scripts/run_quantum.py --set "${SET:-qdev}" --method "$method" --tag "$tag" \
      --procs "${PROCS:-4}" --out "${OUT:-results/qdev.jsonl}" ${COMMON:-} $extra \
      2>&1 | grep -v -E 'cuda|Source Location|external/|^E1[0-9]{3}|^$' || true
done < "$JOBS"
