#!/usr/bin/env bash
# Re-run every method on the 25 Q-set chains used for VQE (every 4th chain), at the
# same 100 / 1000 / 10000 shot budgets and seeds as the main Q-set run, recording sensitivity
# and PPV for every readout (needed for the method-comparison bar figure).
set -euo pipefail
cd "$(dirname "$0")/.."
SET=qset OUT=results/method_comparison.jsonl COMMON="--every 4 --shots 100,1000,10000" \
    bash scripts/run_jobs.sh jobs/method_comparison.txt
echo ALL_DONE
