#!/usr/bin/env bash
# Follow-up runs: constrained drive, warm start, VQE, noise trajectories, scaling.
set -euo pipefail
cd "$(dirname "$0")/.."
S="--shots 100,1000,10000"
SET=qset OUT=results/qset.jsonl COMMON="--seeds 5 $S" \
    bash scripts/run_jobs.sh jobs/qset2.txt
SET=qset OUT=results/vqe.jsonl COMMON="--seeds 1 --every 4 $S" \
    bash scripts/run_jobs.sh jobs/qset_vqe.txt
SET=qset OUT=results/traj.jsonl COMMON="--seeds 2 --every 4 $S" \
    bash scripts/run_jobs.sh jobs/qset_traj.txt
SET=qset OUT=results/scale.jsonl COMMON="--seeds 3 --every 3 --budget 26 --min-stems 26 $S" \
    bash scripts/run_jobs.sh jobs/scale26.txt
SET=qset OUT=results/scale.jsonl PROCS=1 NUMBA_NUM_THREADS=32 \
    COMMON="--seeds 3 --every 6 --budget 28 --min-stems 28 $S" \
    bash scripts/run_jobs.sh jobs/scale28.txt
echo ALL_DONE
