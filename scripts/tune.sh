#!/usr/bin/env bash
# Q-dev tuning: SKQD grid, then LR-QAOA grid plus random / SA controls.
set -euo pipefail
cd "$(dirname "$0")/.."
SET=qdev OUT=results/qdev.jsonl bash scripts/run_jobs.sh jobs/tune_skqd.txt
SET=qdev OUT=results/qdev.jsonl bash scripts/run_jobs.sh jobs/tune_lrqaoa.txt
