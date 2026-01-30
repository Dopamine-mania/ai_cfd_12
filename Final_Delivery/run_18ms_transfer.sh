#!/usr/bin/env bash
set -euo pipefail

# Transfer learning: 26m/s -> 18m/s (Time-Conditioned ResNet one-shot)
# Usage:
#   bash time_conditioned/run_18ms_transfer.sh
#
# Notes:
# - This script does NOT train from scratch. It fine-tunes from the 26m/s weights.
# - Override any env var below on the command line if needed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/processed_data/18ms}"
export SAVE_DIR="${SAVE_DIR:-${REPO_ROOT}/checkpoints_time_18ms/ft_lr1e-5_nf30}"
export FINETUNE_FROM="${FINETUNE_FROM:-${REPO_ROOT}/checkpoints_time/time_resnet_final.pth}"

export EPOCHS="${EPOCHS:-30}"
export LR="${LR:-1e-5}"
export NORM_FACTOR="${NORM_FACTOR:-30.0}"

# A40-friendly default (adjust if OOM):
export BATCH_SIZE="${BATCH_SIZE:-512}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PIN_MEMORY="${PIN_MEMORY:-1}"

mkdir -p "${SAVE_DIR}"
mkdir -p "${REPO_ROOT}/results_time_18ms/final"

echo "[1/2] Fine-tuning on 18m/s data..."
python "${REPO_ROOT}/time_conditioned/train_time_resnet.py"

echo "[2/2] Generating 9s (4500 steps) deliverables..."
export MODEL_PATH="${MODEL_PATH:-${SAVE_DIR}/time_resnet_final.pth}"
export DATA_PATH="${DATA_PATH:-${REPO_ROOT}/processed_data/18ms/340.npy}"
export SAVE_GIF="${SAVE_GIF:-${REPO_ROOT}/results_time_18ms/final/prediction_18ms.gif}"
export SAVE_ERROR_CURVE="${SAVE_ERROR_CURVE:-${REPO_ROOT}/results_time_18ms/final/error_curve_18ms.png}"

export PRED_STEPS="${PRED_STEPS:-4500}"
export GIF_STRIDE="${GIF_STRIDE:-25}"
export BATCH_TIMES="${BATCH_TIMES:-128}"

python "${REPO_ROOT}/time_conditioned/predict_time_resnet.py"

echo "Done."
