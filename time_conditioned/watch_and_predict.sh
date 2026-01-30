#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/checkpoints_time/time_resnet_final.pth}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PRED_LOG="${PRED_LOG:-$LOG_DIR/time_resnet_predict_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$LOG_DIR" "$ROOT_DIR/results_time"

echo "Waiting for Time-Conditioned ResNet weights: $MODEL_PATH"
START_TS="$(date +%s)"
while [[ ! -f "$MODEL_PATH" ]]; do
  sleep 30
done

# 如果文件已存在（例如之前的 smoke test），等待它被“新训练”覆盖（mtime 更新到脚本启动之后）
while true; do
  MTIME="$(stat -c %Y "$MODEL_PATH" 2>/dev/null || echo 0)"
  if [[ "$MTIME" -ge "$START_TS" ]]; then
    break
  fi
  echo "Found old weights (mtime=$MTIME < start=$START_TS), waiting for update..."
  sleep 30
done

echo "Found $MODEL_PATH, running one-shot 9s prediction..."
cd "$ROOT_DIR"
python -u time_conditioned/predict_time_resnet.py 2>&1 | tee "$PRED_LOG"
