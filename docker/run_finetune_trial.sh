#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-gr00t-finetune}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.finetune}"
BUILD_IMAGE="${BUILD_IMAGE:-1}"
CODE_PATH="${CODE_PATH:-$(pwd)}"
CONTAINER_CODE_PATH="${CONTAINER_CODE_PATH:-/workspace}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.7-3B}"
DATASET_PATH="${DATASET_PATH:-demo_data/cube_to_bowl_5}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/docker_finetune_trial}"
MODEL_SOURCE="${MODEL_SOURCE:-}"
SMOKE_MODE="${SMOKE_MODE:-0}"

if [ "$SMOKE_MODE" = "1" ]; then
    MAX_STEPS="${MAX_STEPS:-1}"
    SAVE_STEPS="${SAVE_STEPS:-1}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
else
    MAX_STEPS="${MAX_STEPS:-20}"
    SAVE_STEPS="${SAVE_STEPS:-20}"
    GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
    DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CODE_PATH/$OUTPUT_DIR"

if [ "$BUILD_IMAGE" = "1" ]; then
    DOCKER_BUILDKIT=1 docker build --network host -f "$DOCKERFILE" -t "$IMAGE_NAME" .
fi

TTY_ARGS=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_ARGS=(-it)
fi

docker run --rm "${TTY_ARGS[@]}" --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
    -e HF_HOME=/root/.cache/huggingface \
    -e MODELSCOPE_CACHE=/root/.cache/modelscope \
    -e GROOT_MODEL_SOURCE="$MODEL_SOURCE" \
    -e PYTHONPATH="$CONTAINER_CODE_PATH" \
    -e WANDB_MODE=disabled \
    -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
    -v "${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}:/root/.cache/modelscope" \
    -v "$CODE_PATH:$CONTAINER_CODE_PATH" \
    -w "$CONTAINER_CODE_PATH" \
    "$IMAGE_NAME" \
    bash -lc "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0} python gr00t/experiment/launch_finetune.py \
        --base-model-path '$BASE_MODEL_PATH' \
        --dataset-path '$DATASET_PATH' \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path examples/SO100/so100_config.py \
        --num-gpus 1 \
        --output-dir '$OUTPUT_DIR' \
        --save-total-limit 1 \
        --save-steps '$SAVE_STEPS' \
        --max-steps '$MAX_STEPS' \
        --global-batch-size '$GLOBAL_BATCH_SIZE' \
        --dataloader-num-workers '$DATALOADER_NUM_WORKERS' \
        --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08"
