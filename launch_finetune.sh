#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export GROOT_MODEL_SOURCE="${GROOT_MODEL_SOURCE:-modelscope}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="${HF_HOME:-/mnt/gr00t/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-/mnt/gr00t/cache/modelscope}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATA_NAME="${DATA_NAME:-pnp_wulong_cleaned_v2}"
DATASET_PATH="${DATASET_PATH:-/mnt/gr00t/data/${DATA_NAME}}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-UNITREE_G1_SONIC}"
MODALITY_CONFIG_PATH="${MODALITY_CONFIG_PATH:-gr00t/configs/data/embodiment_configs.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/gr00t/output/${DATA_NAME}}"

SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
MAX_STEPS="${MAX_STEPS:-20000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

LAUNCH_CMD=(
    gr00t/experiment/launch_finetune.py
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$DATASET_PATH"
    --embodiment-tag "$EMBODIMENT_TAG"
    --modality-config-path "$MODALITY_CONFIG_PATH"
    --num-gpus "$NUM_GPUS"
    --output-dir "$OUTPUT_DIR"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --save-steps "$SAVE_STEPS"
    --max-steps "$MAX_STEPS"
    --global-batch-size "$GLOBAL_BATCH_SIZE"
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
    --dataloader-num-workers "$DATALOADER_NUM_WORKERS"
)

if [ "${USE_WANDB:-0}" = "1" ]; then
    LAUNCH_CMD+=(--use-wandb)
fi

if [ -n "${EXTRA_FINETUNE_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=( $EXTRA_FINETUNE_ARGS )
    LAUNCH_CMD+=("${EXTRA_ARGS[@]}")
fi

if [ "$NUM_GPUS" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    exec python "${LAUNCH_CMD[@]}"
fi

exec torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "${LAUNCH_CMD[@]}"
