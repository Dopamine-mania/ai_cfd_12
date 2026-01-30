#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints_fno/fno_final.pth}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PRED_LOG="${PRED_LOG:-$LOG_DIR/fno_predict_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR" "$ROOT_DIR/results_fno"

echo "Waiting for FNO final weights: $MODEL_PATH"
while [[ ! -f "$MODEL_PATH" ]]; do
  sleep 30
done

echo "Found $MODEL_PATH, running prediction..."
cd "$ROOT_DIR"
python -u fno/predict_fno.py 2>&1 | tee "$PRED_LOG"

