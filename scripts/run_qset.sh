#!/usr/bin/env bash
# Final Q-set run with the locked settings (jobs/qset.txt): 5 seeds,
# 100 / 1000 / 10000 shots.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=qset OUT=results/qset.jsonl COMMON="--seeds 5 --shots 100,1000,10000" \
    bash scripts/run_jobs.sh jobs/qset.txt
