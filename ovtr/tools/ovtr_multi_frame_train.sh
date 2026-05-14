#!/bin/sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-9982}"
NPROC_GPU="${NPROC_GPU:-4}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-../model_zoo/ovtr_det_pretrain.pth}"
OUTPUT="${OUTPUT:-./weights}"
CONFIG_FILE="${CONFIG_FILE:-./config/ovtr_5_frame_train_val.py}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

printf 'CONFIG_FILE=%s\n' "${CONFIG_FILE}"
printf 'OUTPUT=%s\n' "${OUTPUT}"
if [ -n "${EXTRA_ARGS}" ]; then
    printf 'EXTRA_ARGS=%s\n' "${EXTRA_ARGS}"
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
    ./main.py \
    --config_file "${CONFIG_FILE}" \
    --dataset_file lvis_generated_img_seqs \
    --epochs 1 \
    --with_box_refine \
    --two_stage \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --lr_drop 13 \
    --pretrain "${PRETRAIN_MODEL}" \
    --num_workers 4 \
    --batch_size "${BATCH_SIZE}" \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --max_len 250 \
    --output_dir "${OUTPUT}" \
    ${EXTRA_ARGS}

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
    ./main.py \
    --config_file "${CONFIG_FILE}" \
    --dataset_file lvis_generated_img_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --resume "${OUTPUT}/checkpoint0000.pth" \
    --num_workers 4 \
    --batch_size "${BATCH_SIZE}" \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --max_len 250 \
    --output_dir "${OUTPUT}" \
    ${EXTRA_ARGS}
