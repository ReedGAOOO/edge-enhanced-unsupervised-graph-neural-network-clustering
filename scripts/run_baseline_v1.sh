#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cora}"
SEED="${2:-0}"
GPU="${3:-0}"

cd "$(dirname "$0")/.."
python3 tools/run_preset.py \
  --preset baseline_v1 \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --gpu "$GPU"
