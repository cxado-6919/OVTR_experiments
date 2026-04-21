#!/bin/sh
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_VARIANT="${MODEL_VARIANT:-lite}"
QUANT_MODE="${QUANT_MODE:-ptq}"
# Supported partitions: exp_a, exp_a1, exp_a2, exp_a3, exp_a4, exp_b
QUANT_PARTITION="${QUANT_PARTITION:-exp_a}"
MASTER_PORT="${MASTER_PORT:-9987}"
OUTPUT="${OUTPUT:-./results_quant}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CALIB_SAMPLES="${CALIB_SAMPLES:-512}"
QAT_EPOCHS="${QAT_EPOCHS:-1}"
QAT_LR="${QAT_LR:-4e-5}"
QAT_LR_BACKBONE="${QAT_LR_BACKBONE:-4e-6}"
QUANT_USE_SCHEDULER="${QUANT_USE_SCHEDULER:-0}"
QUANT_CALIBRATION_ONLY="${QUANT_CALIBRATION_ONLY:-0}"

if [ "${QUANT_MODE}" = "qat" ]; then
    CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
else
    CUDA_DEVICES="${CUDA_DEVICES:-0}"
fi

if [ -z "${NPROC_GPU:-}" ]; then
    IFS=',' read -r -a _ovtr_cuda_devices <<< "${CUDA_DEVICES}"
    NPROC_GPU="${#_ovtr_cuda_devices[@]}"
fi

case "${MODEL_VARIANT}" in
    lite)
        CONFIG_FILE="./config/ovtr_lite_train_val.py"
        PRETRAIN_MODEL="${PRETRAIN_MODEL:-../model_zoo/ovtr_lite.pth}"
        RESULT_PATH="${RESULT_PATH:-./results/teta_results_lite_quant_${QUANT_MODE}_${QUANT_PARTITION}}"
        VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_lite_quant_${QUANT_MODE}_${QUANT_PARTITION}}"
        SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        FILTER_SCORE_THRESH="0.19 0.19 0.19 0.19 0.19 0.19 0.19"
        IOUS_THRESH="0.45 0.45 0.45 0.45 0.45 0.45 0.45"
        ;;
    5_frame)
        CONFIG_FILE="./config/ovtr_5_frame_train_val.py"
        PRETRAIN_MODEL="${PRETRAIN_MODEL:-../model_zoo/ovtr_5_frame.pth}"
        RESULT_PATH="${RESULT_PATH:-./results/teta_results_5_frame_quant_${QUANT_MODE}_${QUANT_PARTITION}}"
        VIS_OUTPUT="${VIS_OUTPUT:-./results/vis_output_track_5_frame_quant_${QUANT_MODE}_${QUANT_PARTITION}}"
        SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        FILTER_SCORE_THRESH="0.20 0.17 0.17 0.20 0.17 0.20 0.17"
        IOUS_THRESH="0.5 0.45 0.5 0.4 0.45 0.45 0.45"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT: ${MODEL_VARIANT}" >&2
        exit 1
        ;;
esac

COMMON_ARGS=(
    --config_file "${CONFIG_FILE}"
    --dataset_file lvis_generated_img_seqs
    --with_box_refine
    --two_stage
    --pretrain "${PRETRAIN_MODEL}"
    --output_dir "${OUTPUT}"
    --num_workers "${NUM_WORKERS}"
    --batch_size "${BATCH_SIZE}"
    --sample_mode random_interval
    --sample_interval 1
    --sampler_steps 4 7 14
    --sampler_lengths 2 3 4 5
    --merger_dropout 0
    --random_drop 0.1
    --fp_ratio 0.3
    --track_query_iteration CIP
    --calculate_negative_samples
    --quant_mode "${QUANT_MODE}"
    --quant_partition "${QUANT_PARTITION}"
    --quant_calib_samples "${CALIB_SAMPLES}"
)

if [ "${QUANT_MODE}" = "qat" ]; then
    CMD=(
        ./main.py
        --epochs "${QAT_EPOCHS}"
        --lr "${QAT_LR}"
        --lr_backbone "${QAT_LR_BACKBONE}"
        --lr_drop 13
        --max_len 250
    )
    if [ "${QUANT_USE_SCHEDULER}" = "1" ]; then
        CMD+=(--quant_use_scheduler)
    fi
    if [ "${QUANT_CALIBRATION_ONLY}" = "1" ]; then
        CMD+=(--quant_calibration_only)
    fi
elif [ "${QUANT_MODE}" = "ptq" ]; then
    CMD=(
        ./eval.py
        --epochs 16
        --lr 4e-5
        --lr_backbone 4e-6
        --lr_drop 13
        --score_thresh ${SCORE_THRESH}
        --filter_score_thresh ${FILTER_SCORE_THRESH}
        --ious_thresh ${IOUS_THRESH}
        --miss_tolerance 5 5 5 5 5 5 5
        --maximum_quantity 160
        --result_path_track "${RESULT_PATH}"
        --vis_output "${VIS_OUTPUT}"
    )
    if [ "${QUANT_CALIBRATION_ONLY}" = "1" ]; then
        CMD+=(--quant_calibration_only)
    fi
else
    echo "Unsupported QUANT_MODE: ${QUANT_MODE}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NPROC_GPU}" \
    "${CMD[@]}" \
    "${COMMON_ARGS[@]}" \
    "$@"
