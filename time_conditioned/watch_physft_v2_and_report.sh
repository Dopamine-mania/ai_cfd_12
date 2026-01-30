#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_DIR="${SAVE_DIR:-$ROOT_DIR/checkpoints_time_phys_v2}"
BEFORE_PATH_FILE="${BEFORE_PATH_FILE:-$SAVE_DIR/BEFORE_PATH.txt}"
AFTER_MODEL_PATH="${AFTER_MODEL_PATH:-$SAVE_DIR/time_resnet_final.pth}"

RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/results_time_phys_v2}"
mkdir -p "$RESULT_DIR" "$ROOT_DIR/logs"

if [[ ! -f "$BEFORE_PATH_FILE" ]]; then
  echo "Missing BEFORE_PATH.txt: $BEFORE_PATH_FILE" >&2
  exit 1
fi
BEFORE_MODEL_PATH="$(cat "$BEFORE_PATH_FILE")"

echo "Waiting for v2 finetuned weights: $AFTER_MODEL_PATH"
while [[ ! -f "$AFTER_MODEL_PATH" ]]; do
  sleep 30
done

echo "Found v2 finetuned weights, generating outputs..."

MODEL_PATH="$AFTER_MODEL_PATH" \
SAVE_GIF="$RESULT_DIR/prediction_9s.gif" \
SAVE_ERROR_CURVE="$RESULT_DIR/error_curve_9s.png" \
python -u time_conditioned/predict_time_resnet.py

BEFORE_MODEL_PATH="$BEFORE_MODEL_PATH" \
AFTER_MODEL_PATH="$AFTER_MODEL_PATH" \
SAVE_FIG="$RESULT_DIR/triptych_step4500.png" \
STEP_IDX=4500 \
COMPONENT=u \
python -u time_conditioned/compare_triptych.py

BEFORE_MODEL_PATH="$BEFORE_MODEL_PATH" \
AFTER_MODEL_PATH="$AFTER_MODEL_PATH" \
SAVE_GIF="$RESULT_DIR/triptych_9s.gif" \
COMPONENT=u \
python -u time_conditioned/compare_triptych_gif.py

echo "Done:"
echo " - $RESULT_DIR/prediction_9s.gif"
echo " - $RESULT_DIR/error_curve_9s.png"
echo " - $RESULT_DIR/triptych_step4500.png"
echo " - $RESULT_DIR/triptych_9s.gif"

