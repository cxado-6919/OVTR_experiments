#!/bin/sh
set -eu

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-9981}"
NPROC_GPU="${NPROC_GPU:-4}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-../model_zoo/ovtr_det_pretrain.pth}"
OUTPUT="${OUTPUT:-./weights}"
CONFIG_FILE="${CONFIG_FILE:-./config/ovtr_lite_train_val.py}"
OMP_THREADS="${OMP_NUM_THREADS:-1}"
NCCL_P2P_DISABLE_VAL="${NCCL_P2P_DISABLE:-1}"
NCCL_SHM_DISABLE_VAL="${NCCL_SHM_DISABLE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-1}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-16}"
STAGE1_LR="${STAGE1_LR:-2e-4}"
STAGE1_LR_BACKBONE="${STAGE1_LR_BACKBONE:-2e-5}"
STAGE2_LR="${STAGE2_LR:-4e-5}"
STAGE2_LR_BACKBONE="${STAGE2_LR_BACKBONE:-4e-6}"
if [ "${STAGE1_EPOCHS}" -lt 1 ]; then
    echo "STAGE1_EPOCHS must be >= 1"
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" OMP_NUM_THREADS="${OMP_THREADS}" NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_VAL}" NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE_VAL}" \
torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
    ./main.py \
    --config_file "${CONFIG_FILE}" \
    --dataset_file lvis_generated_img_seqs \
    --epochs "${STAGE1_EPOCHS}" \
    --with_box_refine \
    --two_stage \
    --lr "${STAGE1_LR}" \
    --lr_backbone "${STAGE1_LR_BACKBONE}" \
    --lr_drop 13 \
    --pretrained "${PRETRAIN_MODEL}" \
    --num_workers "${NUM_WORKERS}" \
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
    "$@"

if [ "${STAGE2_EPOCHS}" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" OMP_NUM_THREADS="${OMP_THREADS}" NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_VAL}" NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE_VAL}" \
    torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
        ./main.py \
        --config_file "${CONFIG_FILE}" \
        --dataset_file lvis_generated_img_seqs \
        --epochs "${STAGE2_EPOCHS}" \
        --with_box_refine \
        --two_stage \
        --lr "${STAGE2_LR}" \
        --lr_backbone "${STAGE2_LR_BACKBONE}" \
        --lr_drop 13 \
        --resume "${OUTPUT}/checkpoint0000.pth" \
        --num_workers "${NUM_WORKERS}" \
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
        "$@"
fi
