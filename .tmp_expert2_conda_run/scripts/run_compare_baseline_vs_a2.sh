#!/usr/bin/env bash
set -euo pipefail

DATASETS="${1:-cora,citeseer,pubmed}"
SEEDS="${2:-0,1,2}"
GPU="${3:-0}"

cd "$(dirname "$0")/.."
python3 tools/compare_baseline_vs_a2.py \
  --datasets "$DATASETS" \
  --seeds "$SEEDS" \
  --gpu "$GPU"
