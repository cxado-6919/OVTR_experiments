#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

CUDA_DEVICES="${CUDA_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-9984}"
NPROC_GPU="${NPROC_GPU:-1}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-../model_zoo/ovtr_5_frame.pth}"
OUTPUT="${OUTPUT:-./results}"
VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_5_frame_val}"
RESULT_PATH="${RESULT_PATH:-./results/teta_results_5_frame_val}"
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
    ./eval.py \
    --config_file ./config/ovtr_5_frame_train_val.py \
    --dataset_file lvis_generated_img_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --pretrain "${PRETRAIN_MODEL}" \
    --output_dir "${OUTPUT}" \
    --num_workers 48 \
    --batch_size 1 \
    --sample_mode random_interval \
    --sample_interval 1 \
    --sampler_steps 4 7 14 \
    --sampler_lengths 2 3 4 5 \
    --merger_dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --track_query_iteration CIP \
    --calculate_negative_samples \
    --score_thresh 0.20 0.17 0.17 0.20 0.17 0.20 0.17 \
    --filter_score_thresh 0.20 0.17 0.17 0.20 0.17 0.20 0.17 \
    --ious_thresh 0.5 0.45 0.5 0.4 0.45 0.45 0.45 \
    --miss_tolerance 5 5 5 5 5 5 5 \
    --maximum_quantity 160 \
    --result_path_track "${RESULT_PATH}" \
    --vis_output "${VIS_OUTPUT}"