#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_VARIANT="${MODEL_VARIANT:-lite}"
EVAL_SPLIT="${EVAL_SPLIT:-val}"
QUANT_PARTITION="${QUANT_PARTITION:-exp_a}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
MASTER_PORT="${MASTER_PORT:-9988}"
NPROC_GPU="${NPROC_GPU:-1}"
OUTPUT="${OUTPUT:-./results}"
NUM_WORKERS="${NUM_WORKERS:-48}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./weights_qat_${MODEL_VARIANT}_${QUANT_PARTITION}}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-${CHECKPOINT_DIR}/checkpoint.pth}"

case "${MODEL_VARIANT}:${EVAL_SPLIT}" in
    lite:val)
        CONFIG_FILE="./config/ovtr_lite_train_val.py"
        RESULT_PATH="${RESULT_PATH:-./results/teta_results_lite_qat_${QUANT_PARTITION}_val}"
        VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_lite_qat_${QUANT_PARTITION}_val}"
        SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        FILTER_SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        IOUS_THRESH="0.45 0.45 0.45 0.45 0.45 0.45 0.45"
        ;;
    lite:test)
        CONFIG_FILE="./config/ovtr_lite_test.py"
        RESULT_PATH="${RESULT_PATH:-./results/teta_results_lite_qat_${QUANT_PARTITION}_test}"
        VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_lite_qat_${QUANT_PARTITION}_test}"
        SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        FILTER_SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        IOUS_THRESH="0.45 0.45 0.45 0.45 0.45 0.45 0.45"
        ;;
    5_frame:val)
        CONFIG_FILE="./config/ovtr_5_frame_train_val.py"
        RESULT_PATH="${RESULT_PATH:-./results/teta_results_5_frame_qat_${QUANT_PARTITION}_val}"
        VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_5_frame_qat_${QUANT_PARTITION}_val}"
        SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        FILTER_SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        IOUS_THRESH="0.5 0.45 0.5 0.4 0.45 0.45 0.45"
        ;;
    5_frame:test)
        CONFIG_FILE="./config/ovtr_5_frame_test.py"
        RESULT_PATH="${RESULT_PATH:-./results/teta_results_5_frame_qat_${QUANT_PARTITION}_test}"
        VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_5_frame_qat_${QUANT_PARTITION}_test}"
        SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        FILTER_SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        IOUS_THRESH="0.5 0.45 0.5 0.4 0.45 0.45 0.45"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT/EVAL_SPLIT combination: ${MODEL_VARIANT}/${EVAL_SPLIT}" >&2
        exit 1
        ;;
esac

if [ ! -f "${PRETRAIN_MODEL}" ]; then
    echo "QAT checkpoint not found: ${PRETRAIN_MODEL}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
    ./eval.py \
    --config_file "${CONFIG_FILE}" \
    --dataset_file lvis_generated_img_seqs \
    --epochs 16 \
    --with_box_refine \
    --two_stage \
    --lr 4e-5 \
    --lr_backbone 4e-6 \
    --lr_drop 13 \
    --pretrain "${PRETRAIN_MODEL}" \
    --output_dir "${OUTPUT}" \
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
    --quant_mode qat \
    --quant_partition "${QUANT_PARTITION}" \
    --score_thresh ${SCORE_THRESH} \
    --filter_score_thresh ${FILTER_SCORE_THRESH} \
    --ious_thresh ${IOUS_THRESH} \
    --miss_tolerance 5 5 5 5 5 5 5 \
    --maximum_quantity 160 \
    --result_path_track "${RESULT_PATH}" \
    --vis_output "${VIS_OUTPUT}" \
    "$@"
